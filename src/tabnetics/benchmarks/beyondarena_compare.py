"""BeyondArena result normalization and paired comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

import io
import math

import numpy as np
import pandas as pd
import scipy.stats as sps


OFFICIAL_BEYONDARENA_METHODS: Tuple[str, ...] = (
    "Linear/Logistic Regression",
    "Random Forest",
    "ExtraTrees",
    "CatBoost",
    "LightGBM",
    "XGBoost",
    "RealMLP",
    "TabM",
    "TabDPT",
    "TabPFN-2.6",
    "TabICLv2",
)
OFFICIAL_BEST_TFM_METHOD = "official_best_tfm"
OFFICIAL_TFM_METHODS: Tuple[str, ...] = ("TabDPT", "TabPFN-2.6", "TabICLv2")
PUBLIC_R2_BEYONDARENA_RESULTS_SOURCE = "public-r2"
PUBLIC_R2_BEYONDARENA_BASE_URL = "https://data.tabarena.ai/cache/artifacts/beyond_iid_benchmark_2026/methods"
BEYONDARENA_METRIC_CONTRACT_VERSION = "beyondarena_metric_contract_v2"

_PUBLIC_R2_METHODS: Tuple[Tuple[str, str, str], ...] = (
    ("LinearModel", "Linear/Logistic Regression", "LinearModel_c1_BAG_L1"),
    ("RandomForest", "Random Forest", "RandomForest_c1_BAG_L1"),
    ("ExtraTrees", "ExtraTrees", "ExtraTrees_c1_BAG_L1"),
    ("CatBoost", "CatBoost", "CatBoost_c1_BAG_L1"),
    ("LightGBM", "LightGBM", "LightGBM_c1_BAG_L1"),
    ("XGBoost", "XGBoost", "XGBoost_c1_BAG_L1"),
    ("TA-RealMLP", "RealMLP", "TA-RealMLP_c1_BAG_L1"),
    ("TA-TabM", "TabM", "TA-TabM_c1_BAG_L1"),
    ("TA-TabDPT", "TabDPT", "TA-TabDPT_c1_BAG_L1"),
    ("TA-TabPFN-2.6", "TabPFN-2.6", "TA-TabPFN-2.6_c1_BAG_L1"),
    ("TA-TabICLv2", "TabICLv2", "TA-TabICLv2_c1_BAG_L1"),
)
_PUBLIC_R2_METHOD_BY_PAPER_NAME = {paper: (path, default) for path, paper, default in _PUBLIC_R2_METHODS}

_METHOD_ALIASES: Dict[str, str] = {
    "linear": "Linear/Logistic Regression",
    "linearmodel": "Linear/Logistic Regression",
    "linearmodel c1 bag l1": "Linear/Logistic Regression",
    "linearmodel_c1_bag_l1": "Linear/Logistic Regression",
    "linear/logistic regression": "Linear/Logistic Regression",
    "logistic regression": "Linear/Logistic Regression",
    "lr": "Linear/Logistic Regression",
    "rf": "Random Forest",
    "randomforest": "Random Forest",
    "randomforest c1 bag l1": "Random Forest",
    "randomforest_c1_bag_l1": "Random Forest",
    "random forest": "Random Forest",
    "extratrees": "ExtraTrees",
    "extratrees c1 bag l1": "ExtraTrees",
    "extratrees_c1_bag_l1": "ExtraTrees",
    "extra trees": "ExtraTrees",
    "extra_tree": "ExtraTrees",
    "cat": "CatBoost",
    "catboost": "CatBoost",
    "catboost c1 bag l1": "CatBoost",
    "catboost_c1_bag_l1": "CatBoost",
    "gbm": "LightGBM",
    "lightgbm": "LightGBM",
    "lightgbm c1 bag l1": "LightGBM",
    "lightgbm_c1_bag_l1": "LightGBM",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "xgboost": "XGBoost",
    "xgboost c1 bag l1": "XGBoost",
    "xgboost_c1_bag_l1": "XGBoost",
    "realmlp": "RealMLP",
    "realmlp_td": "RealMLP",
    "realmlp td": "RealMLP",
    "ta-realmlp": "RealMLP",
    "ta realmlp": "RealMLP",
    "ta-realmlp c1 bag l1": "RealMLP",
    "ta-realmlp_c1_bag_l1": "RealMLP",
    "tabm": "TabM",
    "tabm_official": "TabM",
    "tabm official": "TabM",
    "ta-tabm": "TabM",
    "ta tabm": "TabM",
    "ta-tabm c1 bag l1": "TabM",
    "ta-tabm_c1_bag_l1": "TabM",
    "tabdpt": "TabDPT",
    "ta-tabdpt": "TabDPT",
    "ta tabdpt": "TabDPT",
    "ta-tabdpt c1 bag l1": "TabDPT",
    "ta-tabdpt_c1_bag_l1": "TabDPT",
    "tabpfn": "TabPFN-2.6",
    "tabpfn-2.6": "TabPFN-2.6",
    "tabpfn2.6": "TabPFN-2.6",
    "ta-tabpfn-2.6": "TabPFN-2.6",
    "ta tabpfn-2.6": "TabPFN-2.6",
    "ta-tabpfn-2.6 c1 bag l1": "TabPFN-2.6",
    "ta-tabpfn-2.6_c1_bag_l1": "TabPFN-2.6",
    "tabiclv2": "TabICLv2",
    "tabicl": "TabICLv2",
    "ta-tabiclv2": "TabICLv2",
    "ta tabiclv2": "TabICLv2",
    "ta-tabiclv2 c1 bag l1": "TabICLv2",
    "ta-tabiclv2_c1_bag_l1": "TabICLv2",
}

_LOWER_IS_BETTER_METRICS = {
    "log_loss",
    "mae",
    "mse",
    "metric_error",
    "normalized_error",
    "normalized-error",
    "root_mean_squared_error",
    "rmse",
}
_HIGHER_IS_BETTER_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "amex_metric",
    "r2",
    "f1",
    "macro_f1",
}
_METRIC_ALIASES = {
    "auc": "roc_auc",
    "neg_logloss": "log_loss",
    "logloss": "log_loss",
    "neg_rmse": "rmse",
    "root_mean_squared_error": "rmse",
}
_SUBGROUP_SOURCES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("task_type", ("task_type_official", "task_type_local", "task_type")),
    ("problem_type", ("problem_type_official", "problem_type_local", "problem_type")),
    ("size_tier", ("size_tier_official", "size_tier_local", "size_tier")),
    ("dimensionality", ("dimensionality_official", "dimensionality_local", "dimensionality")),
    ("has_text", ("has_text_official", "has_text_local", "has_text")),
    (
        "high_cardinality",
        ("high_cardinality_official", "high_cardinality_local", "high_cardinality"),
    ),
)


@dataclass(frozen=True)
class BeyondArenaOfficialResultsStatus:
    """Inventory status for official result artifacts."""

    source: str
    available: bool
    exact_paired_rows_available: bool
    reason: str
    columns: Tuple[str, ...] = ()
    row_count: int = 0


@dataclass(frozen=True)
class BeyondArenaComparisonArtifacts:
    """Outputs from a BeyondArena official-vs-local comparison."""

    official: pd.DataFrame
    local: pd.DataFrame
    joined: pd.DataFrame
    summary: pd.DataFrame
    status: BeyondArenaOfficialResultsStatus


def normalize_beyondarena_method_name(name: Any) -> str:
    """Normalize paper/local method aliases without hiding unknown names."""

    text = str(name or "").strip()
    key = text.lower().replace("_", " ")
    key = " ".join(key.split())
    return _METHOD_ALIASES.get(key, text)


def metric_lower_is_better(metric: Any, *, value_column: str = "metric_value") -> bool:
    key = normalize_beyondarena_metric_name(metric)
    if str(value_column).strip().lower() == "metric_error":
        return True
    if key in _LOWER_IS_BETTER_METRICS:
        return True
    if key in _HIGHER_IS_BETTER_METRICS:
        return False
    return True


def _coerce_metric_direction(
    value: Any,
    *,
    metric: Any,
    value_column: str,
) -> bool:
    """Preserve an explicit metric direction, inferring only when it is absent."""

    if value is None or (not isinstance(value, (list, tuple, dict)) and bool(pd.isna(value))):
        return metric_lower_is_better(metric, value_column=value_column)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and float(value) in {0.0, 1.0}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    raise ValueError(
        f"Invalid lower_is_better value {value!r} for BeyondArena metric {metric!r}"
    )


def _normalize_metric_value_semantics(values: pd.Series) -> pd.Series:
    semantics = values.fillna("").astype(str).str.strip().str.lower()
    invalid = ~semantics.isin({"metric", "score", "error"})
    if bool(invalid.any()):
        bad = sorted(set(semantics.loc[invalid].tolist()))
        raise ValueError(f"Invalid BeyondArena metric_value_semantics values: {bad[:5]}")
    return semantics.replace({"score": "metric"})


def normalize_beyondarena_metric_name(metric: Any) -> str:
    """Normalize equivalent BeyondArena metric spellings to one join key."""

    key = str(metric or "").strip().lower()
    return _METRIC_ALIASES.get(key, key)


def _read_table(source: str | Path) -> pd.DataFrame:
    if isinstance(source, Path) or (isinstance(source, str) and "://" not in source):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    with urlopen(str(source)) as response:
        payload = response.read()
    if str(source).lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(payload))
    return pd.read_csv(io.BytesIO(payload))


def _is_public_r2_source(source: Any) -> bool:
    return str(source or "").strip().lower() in {
        PUBLIC_R2_BEYONDARENA_RESULTS_SOURCE,
        "official-r2",
        "beyondarena-r2",
        "beyondarena-public-r2",
    }


def _read_public_r2_parquet(method_path: str) -> pd.DataFrame:
    from urllib.parse import quote

    url = f"{PUBLIC_R2_BEYONDARENA_BASE_URL}/{quote(method_path, safe='')}/results/model_results.parquet"
    request = Request(
        url,
        headers={"User-Agent": "tabnetics-beyondarena/1.0"},
    )
    with urlopen(request) as response:
        payload = response.read()
    return pd.read_parquet(io.BytesIO(payload))


def load_public_beyondarena_r2_results(
    *,
    methods: Optional[Iterable[str]] = None,
    include_all_configs: bool = False,
) -> pd.DataFrame:
    """Load exact public BeyondArena baseline rows from TabArena's R2 cache.

    By default this returns the upstream configured default row for each method
    (`*_c1_BAG_L1`).  Set ``include_all_configs=True`` for an inventory/debug
    view that includes HPO/config rows for methods that published them.
    """

    wanted = None if methods is None else {normalize_beyondarena_method_name(name) for name in methods}
    frames: list[pd.DataFrame] = []
    for method_path, paper_name, default_config in _PUBLIC_R2_METHODS:
        if wanted is not None and paper_name not in wanted:
            continue
        raw = _read_public_r2_parquet(method_path)
        if not include_all_configs and "method" in raw.columns:
            raw = raw[raw["method"].astype(str).eq(default_config)].copy()
        out = pd.DataFrame()
        out["dataset_id"] = raw["dataset"].astype(str)
        out["split_id"] = raw["fold"].astype(str)
        out["method"] = paper_name
        out["metric"] = raw["metric"].map(normalize_beyondarena_metric_name)
        out["metric_value"] = pd.to_numeric(raw["metric_error"], errors="coerce")
        out["metric_error"] = out["metric_value"]
        out["metric_value_semantics"] = "error"
        out["origin"] = "official_public_r2"
        out["lower_is_better"] = True
        out["status"] = "ok"
        out["official_config"] = raw.get("method", default_config)
        out["ta_suite"] = raw.get("ta_suite", "beyond_iid_benchmark_2026")
        out["ta_name"] = raw.get("ta_name", method_path)
        if "problem_type" in raw.columns:
            out["problem_type"] = raw["problem_type"]
        if "metric_error_val" in raw.columns:
            out["metric_error_val"] = raw["metric_error_val"]
        if "time_train_s" in raw.columns:
            out["time_train_s"] = raw["time_train_s"]
        if "time_infer_s" in raw.columns:
            out["time_infer_s"] = raw["time_infer_s"]
        frames.append(out.reset_index(drop=True))
    if not frames:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "split_id",
                "method",
                "metric",
                "metric_value",
                "metric_error",
                "metric_value_semantics",
                "origin",
                "lower_is_better",
                "status",
            ]
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def _first_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    cols = {str(col).lower(): str(col) for col in df.columns}
    for alias in aliases:
        if alias.lower() in cols:
            return cols[alias.lower()]
    return None


def normalize_beyondarena_result_table(
    df: pd.DataFrame,
    *,
    origin: str,
    default_method: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize official or local rows to the BeyondArena comparison schema."""

    source = df.copy()
    dataset_col = _first_column(source, ("beyondarena_id", "dataset_id", "dataset", "task"))
    method_col = _first_column(source, ("method", "model", "classifier", "profile", "model_name"))
    metric_col = _first_column(source, ("metric", "metric_name", "objective_metric_name"))
    value_col = _first_column(source, ("metric_value", "score", "value", "metric_error", "error"))
    value_semantics_col = _first_column(
        source,
        ("metric_value_semantics", "metric_value_kind", "value_semantics", "value_kind"),
    )
    direction_col = _first_column(source, ("lower_is_better", "metric_lower_is_better"))
    split_col = _first_column(source, ("split_id", "split", "fold_id", "fold"))
    repeat_col = _first_column(source, ("repeat", "repeat_id", "outer_repeat"))
    fold_col = _first_column(source, ("fold", "fold_id", "outer_fold"))

    missing = []
    if dataset_col is None:
        missing.append("dataset")
    if method_col is None and default_method is None:
        missing.append("method")
    if metric_col is None:
        missing.append("metric")
    if value_col is None:
        missing.append("metric_value")
    if missing:
        raise ValueError(f"BeyondArena result table missing required columns: {missing}")

    out = pd.DataFrame()
    out["dataset_id"] = source[dataset_col].astype(str)
    out["method"] = (
        source[method_col].map(normalize_beyondarena_method_name)
        if method_col is not None
        else str(default_method)
    )
    out["metric"] = source[metric_col].map(normalize_beyondarena_metric_name)
    out["metric_value"] = pd.to_numeric(source[value_col], errors="coerce")
    if value_semantics_col is not None:
        out["metric_value_semantics"] = _normalize_metric_value_semantics(
            source[value_semantics_col]
        )
    elif str(value_col).strip().lower() in {"metric_error", "error"}:
        out["metric_value_semantics"] = "error"
    else:
        out["metric_value_semantics"] = "metric"
    if split_col is not None:
        out["split_id"] = source[split_col].astype(str)
    elif repeat_col is not None and fold_col is not None:
        out["split_id"] = source[repeat_col].astype(str) + ":" + source[fold_col].astype(str)
    else:
        out["split_id"] = "0:0"
    out["origin"] = str(origin)
    explicit_direction = (
        source[direction_col].tolist() if direction_col is not None else [None] * len(source)
    )
    out["lower_is_better"] = [
        _coerce_metric_direction(
            declared,
            metric=metric,
            value_column=str(value_col),
        )
        for declared, metric in zip(explicit_direction, out["metric"])
    ]
    invalid_error_direction = out["metric_value_semantics"].eq("error") & ~out[
        "lower_is_better"
    ].astype(bool)
    if bool(invalid_error_direction.any()):
        bad = out.loc[invalid_error_direction, ["dataset_id", "metric"]].to_dict("records")
        raise ValueError(
            "BeyondArena error-valued rows must declare lower_is_better=true: "
            f"{bad[:5]}"
        )

    optional_map = {
        "problem_type": ("problem_type", "task_problem_type"),
        "task_type": ("task_type", "split_family", "subbenchmark"),
        "size_tier": ("size_tier", "size_bin"),
        "dimensionality": ("dimensionality", "dimension_tier"),
        "has_text": ("has_text", "text", "text_features"),
        "high_cardinality": ("high_cardinality", "has_high_cardinality"),
        "preprocessing_profile": ("preprocessing_profile",),
        "model_profile": ("model_profile",),
        "model_name": ("model_name",),
        "seed": ("seed",),
        "device": ("device",),
        "execution_host": ("execution_host", "host"),
        "execution_lane": ("execution_lane",),
        "execution_status": ("execution_status",),
        "execution_backend": ("execution_backend", "backend"),
        "allow_gpu_execution": ("allow_gpu_execution",),
        "skip_reason": ("skip_reason", "reason"),
        "status": ("status",),
        "local_dataset_id": ("local_dataset_id",),
        "local_split_id": ("local_split_id",),
        "artifact_revision": ("artifact_revision", "dataset_revision"),
        "official_config": ("official_config", "config", "config_type"),
        "ta_suite": ("ta_suite",),
        "ta_name": ("ta_name",),
        "time_train_s": ("time_train_s", "train_time_s"),
        "time_infer_s": ("time_infer_s", "infer_time_s"),
        "metric_error": ("metric_error", "error"),
        "metric_error_val": ("metric_error_val",),
    }
    for target, aliases in optional_map.items():
        col = _first_column(source, aliases)
        if col is not None:
            out[target] = (
                pd.to_numeric(source[col], errors="coerce")
                if target in {"metric_error", "metric_error_val"}
                else source[col]
            )
    if "status" not in out.columns:
        out["status"] = "ok"
    return out.reset_index(drop=True)


def load_official_beyondarena_results(source: str | Path) -> pd.DataFrame:
    """Load and normalize exact official per-dataset/per-split rows."""

    if _is_public_r2_source(source):
        return load_public_beyondarena_r2_results()
    return normalize_beyondarena_result_table(_read_table(source), origin="official")


def inspect_official_beyondarena_results(source: Optional[str | Path] = None) -> BeyondArenaOfficialResultsStatus:
    """Return an explicit status for official result availability."""

    if source is None:
        return BeyondArenaOfficialResultsStatus(
            source="not configured",
            available=False,
            exact_paired_rows_available=False,
            reason=(
                "No official BeyondArena per-dataset/per-split result artifact was configured. "
                "Paper-level aggregate values can be cited, but exact paired parity claims are blocked."
            ),
        )
    try:
        df = load_official_beyondarena_results(source)
    except Exception as exc:
        return BeyondArenaOfficialResultsStatus(
            source=str(source),
            available=False,
            exact_paired_rows_available=False,
            reason=f"Unable to load official BeyondArena results: {exc}",
        )
    required = {"dataset_id", "split_id", "method", "metric", "metric_value"}
    exact = required.issubset(df.columns) and not df.empty
    return BeyondArenaOfficialResultsStatus(
        source=str(source),
        available=not df.empty,
        exact_paired_rows_available=bool(exact),
        reason="exact paired rows available" if exact else "loaded table lacks exact paired rows",
        columns=tuple(str(col) for col in df.columns),
        row_count=int(len(df)),
    )


def build_tabnetics_beyondarena_rows(
    local_results: pd.DataFrame,
    *,
    method_name: str,
    model_profile: str = "tabnetics",
) -> pd.DataFrame:
    """Normalize local Tabnetics Diakrino rows for BeyondArena comparison."""

    rows = normalize_beyondarena_result_table(
        local_results,
        origin="tabnetics",
        default_method=method_name,
    )
    rows["method"] = str(method_name)
    rows["model_profile"] = str(model_profile)
    return rows


def append_official_best_tfm_rows(official: pd.DataFrame) -> pd.DataFrame:
    """Append the best official TFM row per dataset/split/metric when present."""

    if official.empty or "method" not in official.columns:
        return official.copy()
    frame = official.copy()
    tfm = frame[frame["method"].isin(OFFICIAL_TFM_METHODS)].copy()
    tfm = tfm[tfm.get("status", "ok").astype(str).str.lower().eq("ok")]
    tfm = tfm[pd.to_numeric(tfm["metric_value"], errors="coerce").notna()]
    if tfm.empty:
        return frame

    lower = (
        pd.Series(
            [
                _coerce_metric_direction(value, metric=metric, value_column="metric_value")
                for value, metric in zip(tfm["lower_is_better"], tfm["metric"])
            ],
            index=tfm.index,
            dtype=bool,
        )
        if "lower_is_better" in tfm.columns
        else pd.Series(True, index=tfm.index, dtype=bool)
    )
    values = pd.to_numeric(tfm["metric_value"], errors="coerce")
    tfm["_best_sort_value"] = np.where(lower, values, -values)
    key_cols = ["dataset_id", "split_id", "metric"]
    best_idx = tfm.groupby(key_cols, dropna=False)["_best_sort_value"].idxmin()
    best = tfm.loc[best_idx].drop(columns=["_best_sort_value"]).copy()
    best["best_tfm_source_method"] = best["method"]
    best["method"] = OFFICIAL_BEST_TFM_METHOD
    return pd.concat([frame, best], ignore_index=True, sort=False)


def join_beyondarena_results(
    official: pd.DataFrame,
    local: pd.DataFrame,
    *,
    official_methods: Optional[Iterable[str]] = None,
    local_methods: Optional[Iterable[str]] = None,
    tie_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Join exact official and local rows by dataset/split/metric."""

    off = official.copy()
    loc = local.copy()
    if official_methods is not None:
        wanted = {normalize_beyondarena_method_name(name) for name in official_methods}
        off = off[off["method"].isin(wanted)]
    if local_methods is not None:
        wanted = {str(name) for name in local_methods}
        loc = loc[loc["method"].isin(wanted)]
    off = off[off.get("status", "ok").astype(str).str.lower().eq("ok")]
    loc = loc[loc.get("status", "ok").astype(str).str.lower().eq("ok")]

    key_cols = ["dataset_id", "split_id", "metric"]
    merged = off.merge(
        loc,
        on=key_cols,
        how="inner",
        suffixes=("_official", "_local"),
    )
    if merged.empty:
        return merged
    official_lower = pd.Series(
        [
            _coerce_metric_direction(value, metric=metric, value_column="metric_value")
            for value, metric in zip(merged["lower_is_better_official"], merged["metric"])
        ],
        index=merged.index,
        dtype=bool,
    )
    local_lower = pd.Series(
        [
            _coerce_metric_direction(value, metric=metric, value_column="metric_value")
            for value, metric in zip(merged["lower_is_better_local"], merged["metric"])
        ],
        index=merged.index,
        dtype=bool,
    )
    official_semantics = (
        _normalize_metric_value_semantics(merged["metric_value_semantics_official"])
        if "metric_value_semantics_official" in merged.columns
        else pd.Series("metric", index=merged.index)
    )
    local_semantics = (
        _normalize_metric_value_semantics(merged["metric_value_semantics_local"])
        if "metric_value_semantics_local" in merged.columns
        else pd.Series("metric", index=merged.index)
    )
    same_direction = official_lower == local_lower
    same_semantics = official_semantics == local_semantics
    merged["comparison_value_official"] = pd.to_numeric(
        merged["metric_value_official"], errors="coerce"
    )
    merged["comparison_value_local"] = pd.to_numeric(
        merged["metric_value_local"], errors="coerce"
    )
    finite_primary_values = pd.Series(
        np.isfinite(merged["comparison_value_official"])
        & np.isfinite(merged["comparison_value_local"]),
        index=merged.index,
    )
    primary_values_comparable = same_direction & same_semantics & finite_primary_values
    merged["comparison_value_semantics"] = official_semantics
    comparison_lower_is_better = official_lower.copy()
    if not bool(primary_values_comparable.all()):
        if {"metric_error_official", "metric_error_local"}.issubset(merged.columns):
            official_error = pd.to_numeric(merged["metric_error_official"], errors="coerce")
            local_error = pd.to_numeric(merged["metric_error_local"], errors="coerce")
            comparable_error = pd.Series(
                np.isfinite(official_error) & np.isfinite(local_error),
                index=merged.index,
            )
        else:
            comparable_error = pd.Series(False, index=merged.index)
        unresolved = ~primary_values_comparable & ~comparable_error
        if bool(unresolved.any()):
            bad = merged.loc[unresolved, key_cols].drop_duplicates().to_dict("records")
            raise ValueError(
                "Metric direction/representation mismatch in BeyondArena comparison rows: "
                f"{bad[:5]}"
            )
        use_error = ~primary_values_comparable & comparable_error
        merged.loc[use_error, "comparison_value_official"] = pd.to_numeric(
            merged.loc[use_error, "metric_error_official"],
            errors="coerce",
        )
        merged.loc[use_error, "comparison_value_local"] = pd.to_numeric(
            merged.loc[use_error, "metric_error_local"],
            errors="coerce",
        )
        merged.loc[use_error, "comparison_value_semantics"] = "error"
        comparison_lower_is_better.loc[use_error] = True

    merged["comparison_lower_is_better"] = comparison_lower_is_better.astype(bool)
    direction = np.where(merged["comparison_lower_is_better"], -1.0, 1.0)
    source_value_delta = (
        merged["metric_value_local"].astype(float) - merged["metric_value_official"].astype(float)
    )
    comparison_value_delta = (
        merged["comparison_value_local"].astype(float) - merged["comparison_value_official"].astype(float)
    )
    # Positive comparison_delta is good for the local method, regardless of metric direction.
    merged["comparison_value_delta"] = comparison_value_delta
    merged["comparison_delta"] = comparison_value_delta * direction
    merged["source_value_delta"] = source_value_delta
    merged["raw_metric_delta"] = source_value_delta.where(primary_values_comparable, np.nan)
    merged["outcome"] = np.where(
        merged["comparison_delta"] > float(tie_tolerance),
        "win",
        np.where(merged["comparison_delta"] < -float(tie_tolerance), "loss", "tie"),
    )
    return merged.reset_index(drop=True)


def summarize_beyondarena_pairs(
    joined: pd.DataFrame,
    *,
    group_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Summarize paired BeyondArena rows with W/T/L and Wilcoxon where possible."""

    if joined.empty:
        cols = [*group_cols, "n_pairs", "wins", "ties", "losses", "mean_delta", "median_delta", "wilcoxon_p"]
        return pd.DataFrame(columns=cols)
    groups = [col for col in group_cols if col in joined.columns]
    if groups:
        iterator = joined.groupby(groups, dropna=False)
    else:
        iterator = (("all", joined),)

    records: list[dict[str, Any]] = []
    for key, df in iterator:
        record: Dict[str, Any] = {}
        if groups:
            if not isinstance(key, tuple):
                key = (key,)
            record.update({col: value for col, value in zip(groups, key)})
        deltas = pd.to_numeric(df["comparison_delta"], errors="coerce").dropna()
        nonzero = deltas[np.abs(deltas) > 1e-12]
        if len(nonzero) >= 1:
            try:
                _stat, p_value = sps.wilcoxon(nonzero, alternative="two-sided")
            except Exception:
                p_value = math.nan
        else:
            p_value = math.nan
        outcomes = df["outcome"].value_counts()
        record.update(
            {
                "n_pairs": int(len(df)),
                "wins": int(outcomes.get("win", 0)),
                "ties": int(outcomes.get("tie", 0)),
                "losses": int(outcomes.get("loss", 0)),
                "mean_delta": float(deltas.mean()) if len(deltas) else math.nan,
                "median_delta": float(deltas.median()) if len(deltas) else math.nan,
                "wilcoxon_p": float(p_value) if p_value == p_value else math.nan,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _first_available_joined_column(joined: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in joined.columns:
            return col
    return None


def _add_summary_subgroup_columns(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return joined.copy()
    out = joined.copy()
    for subgroup, candidates in _SUBGROUP_SOURCES:
        source_col = _first_available_joined_column(out, candidates)
        if source_col is not None:
            out[f"summary_{subgroup}"] = out[source_col]
    return out


def _scoped_summary_frame(
    joined: pd.DataFrame,
    *,
    scope: str,
    group_cols: Sequence[str],
    value_col: Optional[str] = None,
) -> pd.DataFrame:
    summary = summarize_beyondarena_pairs(joined, group_cols=group_cols)
    summary.insert(0, "summary_scope", str(scope))
    if value_col is None:
        summary.insert(1, "summary_value", "all")
    elif value_col in summary.columns:
        summary = summary.rename(columns={value_col: "summary_value"})
    else:
        summary.insert(1, "summary_value", pd.NA)
    return summary


def build_beyondarena_summary_table(joined: pd.DataFrame) -> pd.DataFrame:
    """Build pair-aware overall and subgroup summaries for exact joined rows."""

    if joined.empty:
        return _scoped_summary_frame(joined, scope="all", group_cols=())

    enriched = _add_summary_subgroup_columns(joined)
    pair_cols = [col for col in ("method_official", "method_local") if col in enriched.columns]
    frames = [_scoped_summary_frame(enriched, scope="all", group_cols=pair_cols)]
    for subgroup, _candidates in _SUBGROUP_SOURCES:
        col = f"summary_{subgroup}"
        if col not in enriched.columns:
            continue
        frames.append(
            _scoped_summary_frame(
                enriched,
                scope=subgroup,
                group_cols=(*pair_cols, col),
                value_col=col,
            )
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def build_beyondarena_comparison_artifacts(
    official_results: pd.DataFrame,
    local_results: pd.DataFrame,
    *,
    official_methods: Optional[Iterable[str]] = None,
    local_methods: Optional[Iterable[str]] = None,
) -> BeyondArenaComparisonArtifacts:
    """Build joined rows and subgroup summaries for exact paired comparisons."""

    official = append_official_best_tfm_rows(
        normalize_beyondarena_result_table(official_results, origin="official")
    )
    local = normalize_beyondarena_result_table(local_results, origin="tabnetics")
    joined = join_beyondarena_results(
        official,
        local,
        official_methods=official_methods,
        local_methods=local_methods,
    )
    joined = _add_summary_subgroup_columns(joined)
    summary = build_beyondarena_summary_table(joined)
    status = BeyondArenaOfficialResultsStatus(
        source="dataframe",
        available=not official.empty,
        exact_paired_rows_available=not official.empty,
        reason="exact paired rows available from dataframe",
        columns=tuple(str(col) for col in official.columns),
        row_count=int(len(official)),
    )
    return BeyondArenaComparisonArtifacts(
        official=official,
        local=local,
        joined=joined,
        summary=summary,
        status=status,
    )


__all__ = [
    "BEYONDARENA_METRIC_CONTRACT_VERSION",
    "OFFICIAL_BEST_TFM_METHOD",
    "OFFICIAL_BEYONDARENA_METHODS",
    "OFFICIAL_TFM_METHODS",
    "PUBLIC_R2_BEYONDARENA_BASE_URL",
    "PUBLIC_R2_BEYONDARENA_RESULTS_SOURCE",
    "append_official_best_tfm_rows",
    "BeyondArenaComparisonArtifacts",
    "BeyondArenaOfficialResultsStatus",
    "build_beyondarena_comparison_artifacts",
    "build_beyondarena_summary_table",
    "build_tabnetics_beyondarena_rows",
    "inspect_official_beyondarena_results",
    "join_beyondarena_results",
    "load_official_beyondarena_results",
    "load_public_beyondarena_r2_results",
    "metric_lower_is_better",
    "normalize_beyondarena_metric_name",
    "normalize_beyondarena_method_name",
    "normalize_beyondarena_result_table",
    "summarize_beyondarena_pairs",
]

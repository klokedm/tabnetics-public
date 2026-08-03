"""Leakage-safe loader for persisted DIAKRINO FS-teacher sidecar logits.

The trained Tabentics FS-teacher DIAKRINO emits, per support-only inference, a *sidecar*
parquet of per-feature logits and population-reconstruction head outputs.  This module
reads that sidecar on **CPU only** (no model, no torch) and exposes chunk-calibrated,
leakage-disciplined signals to the opt-in native-integration hooks.

Design contract (see ``research/TABENTICS_DIAKRINO_NATIVE_INTEGRATION.md``):

* **Optional dependency.** ``pandas`` is imported lazily; every public entry point
  returns ``None`` when pandas is unavailable or the sidecar file is missing, so
  callers degrade to current behaviour (mirrors the ``TABPFN_AVAILABLE`` guard in
  ``pipeline.py``).
* **No inference.** This module never loads the DIAKRINO; it only reads a persisted
  artifact emitted (on ``public-gpu-host``) from TRAIN/support rows.  It is therefore safe to
  call from the inference/replay path *only* for signals that are also persisted into
  ``feature_plans`` — see the replay-parity doctrine.  Path-changing hooks must reduce
  their decision to a persisted ``apply_reason`` at fit time.
* **Chunk calibration mandatory.** The only calibration-safe scalar surfaces for any
  global top-k / soft-prior / relevance use are ``prior_logit`` / ``screening_logit``
  (chunk-flat); raw ``base_logit`` / ``feature_selection_logit`` / ``selector_gate_logit``
  are chunk-contaminated and must be within-chunk normalised before cross-chunk use.
* **Family-id discipline.** The 36-way population-family head shares an id space with
  the experimental synthetic-world family vocabulary; only ids ``0..30`` map onto the
  continuous core ``UnifiedDistributionSelectorV6`` family set.  Ids ``>=31`` (discrete
  + nuisance overflow) must be routed to ``skip_fit`` before any continuous decode.
  The byte-identity of this vocabulary with the experimental source is asserted by
  ``core/tests/test_diakrino_sidecar.py`` (tests may import experimental; the runtime module
  here must not).
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .diakrino_identity import (
    DIAKRINO_REQUIRED_CALIBRATION_MODE,
    DiakrinoExpectedIdentity,
    DiakrinoSidecarIdentityError,
    canonical_json_sha256,
    exact_int64_scalar,
    exact_int64_vector,
    read_verified_artifact_reference,
    read_verified_artifacts,
    resolve_manifest_identity_record,
    sha256_bytes,
    validate_frame_identity,
)
from .diakrino_trust import (
    normalize_diakrino_sidecar_trust_record,
    qualification_record_covers_identity,
    source_emission_qualification_covers_identity,
    trust_record_checkpoint_sha256,
    trust_record_discrete_filter_ok,
    trust_record_head_allowed,
    trust_record_required_calibration,
)
from .diakrino_views import (
    DIAKRINO_FROZEN_VIEW_IDS,
    DIAKRINO_VIEW_SCORE_SOURCE,
    DiakrinoViewError,
    validate_view_artifact,
)

# ---------------------------------------------------------------------------
# Canonical population-family vocabulary (id -> name).
#
# Byte-identical to ``experimental.population_ground_truth.MARGINAL_FAMILIES`` and
# ``experimental.worlds.ALL_SUPPORTED_FAMILIES`` (CORE16 + EXT6 + FLEX9 + DISCRETE5).
# Duplicated here deliberately so the core runtime carries no experimental import;
# a regression test asserts the three literals stay identical.  Ids 0..30 are the
# continuous families consumed by the core distribution selector; ids 31..35 are
# discrete; any id >= 31 (incl. nuisance overflow 36) is unmappable -> skip_fit.
# ---------------------------------------------------------------------------
CANONICAL_MARGINAL_FAMILIES: tuple[str, ...] = (
    "norm", "expon", "uniform", "weibull_min", "gamma", "lognorm", "beta", "t",
    "laplace", "pareto", "gumbel_l", "gumbel_r", "powerlaw", "triang", "johnsonsu",
    "johnsonsb", "skewnorm", "genextreme", "invgamma", "fisk", "genpareto", "gengamma",
    "tukeylambda", "gennorm", "genlogistic", "logistic", "moyal", "genhyperbolic",
    "invweibull", "invgauss", "geninvgauss",          # ids 0..30  -> continuous
    "poisson", "neg_binom", "zip", "binary_threshold", "ordinal_threshold",  # 31..35 discrete
)
N_CANONICAL_FAMILIES: int = len(CANONICAL_MARGINAL_FAMILIES)          # 36
N_CONTINUOUS_FAMILIES: int = 31                                       # ids 0..30
CONTINUOUS_FAMILY_IDS: frozenset[int] = frozenset(range(N_CONTINUOUS_FAMILIES))
POPULATION_DEPENDENCY_DIM: int = 8
POPULATION_COEFF_DIM: int = 4
POPULATION_DEPENDENCY_CHANNELS: tuple[str, ...] = (
    "block_id_scaled",
    "block_loading_norm",
    "block_loading_0",
    "block_loading_1",
    "max_abs_cov_or_precision_row",
    "mean_abs_cov_or_precision_row",
    "edge_degree_scaled",
    "canonical_dependence_scalar",
)
POPULATION_COEFF_CHANNELS: tuple[str, ...] = (
    "linear_or_main_effect_weight",
    "interaction_or_higher_order_weight",
    "max_polynomial_order",
    "active_feature_flag",
)
DIAKRINO_SELECTOR_POOL: tuple[str, ...] = (
    "mnpo_broad_stable",
    "strict_plus_mrmr",
    "boruta",
    "copula_knockoff",
    "stability_lasso",
)

# Sidecar ``feature_logits`` scalar columns are SINGULAR-named (one float per feature).
SCALAR_SCORE_COLUMNS: tuple[str, ...] = (
    "feature_selection_logit", "prior_logit", "screening_logit", "series_logit",
    "residual_logit", "selector_gate_logit", "refiner_logit", "class_extras_logit",
)
SCHEMA_V2_SCALAR_COLUMNS: tuple[str, ...] = (
    "population_reconstruction_mse",
    "population_class_reconstruction_mse",
    "conformal_score",
    "conformal_selection_probability",
)
SCHEMA_V2_VECTOR_COLUMNS: tuple[str, ...] = (
    "population_reconstruction_predictions",
    "population_reconstruction_targets",
    "population_reconstruction_residuals",
    "population_class_reconstruction_predictions",
    "population_class_reconstruction_targets",
    "population_class_reconstruction_residuals",
)
# Only these are chunk-flat / safe for cross-chunk global top-k without z-scoring.
DIAKRINO_UNIFORM_VIEW_RANK_COLUMN = "diakrino_uniform_view_rank01"
CALIBRATION_SAFE_COLUMNS: frozenset[str] = frozenset(
    {"prior_logit", "screening_logit", DIAKRINO_UNIFORM_VIEW_RANK_COLUMN}
)
SCHEMA_V2_CALIBRATION_SAFE_COLUMNS: frozenset[str] = frozenset()
SCHEMA_V2_CHUNK_CONTAMINATED_COLUMNS: frozenset[str] = frozenset(SCHEMA_V2_SCALAR_COLUMNS)
SCHEMA_V2_SCALAR_TRUST_HEADS: dict[str, str] = {
    "population_reconstruction_mse": "population_reconstruction",
    "population_class_reconstruction_mse": "population_class_reconstruction",
    "conformal_score": "conformal_selection",
    "conformal_selection_probability": "conformal_selection",
}
RECONSTRUCTION_TRUST_HEADS: dict[str, str] = {
    "population_reconstruction": "population_reconstruction",
    "population_class_reconstruction": "population_class_reconstruction",
}
CALIBRATED_SCORE_SUFFIXES: tuple[str, ...] = (
    "global_rank01",
    "chunk_zscore",
    "chunk_rank01",
    "chunk_ecdf",
    "chunk_minmax",
    "chunk_robust_iqr",
    "chunk_softmax_temp",
    "blend",
    "chunk_zscore_rank01",
    "chunk_robust_iqr_rank01",
    "chunk_softmax_temp_rank01",
)
SELECTION_ALIAS_SUFFIXES: tuple[str, ...] = (
    "zscore",
    "rank01",
    "ecdf",
    "minmax",
    "robust_iqr",
    "softmax_temp",
    "blend",
    "zscore_rank01",
    "robust_iqr_rank01",
    "softmax_temp_rank01",
)
CALIBRATION_MODES: tuple[str, ...] = (
    "none",
    "chunk_zscore",
    "chunk_rank01",
    "chunk_ecdf",
    "chunk_minmax",
    "chunk_robust_iqr",
    "chunk_softmax_temp",
    "blend",
)

_PAD_SENTINEL: float = -30.0   # padded/invalid feature logit sentinel

__tabnetics_execution_isolated_state__ = {
    "N_CANONICAL_FAMILIES": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "N_CONTINUOUS_FAMILIES": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "CONTINUOUS_FAMILY_IDS": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
}


def _import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:        # pragma: no cover - exercised only without pandas
        return None


def _strict_scalar_column_trust_head(column: str) -> str:
    """Return an explicit strict trust head, or ``""`` for unknown columns."""

    name = str(column)
    if name == DIAKRINO_UNIFORM_VIEW_RANK_COLUMN:
        return "feature_selection"
    if name == "selector_weights_candidate" or name.startswith(
        ("selector_weight_candidate_", "selector_weights_candidate_")
    ):
        return "selector_weights_candidate"
    if name == "selector_weights" or name.startswith(
        ("selector_weight_", "selector_weights_")
    ):
        return "selector_weights"
    for base, head in SCHEMA_V2_SCALAR_TRUST_HEADS.items():
        if name == base or any(
            name == f"{base}_{suffix}" for suffix in CALIBRATED_SCORE_SUFFIXES
        ):
            return head
    if name in SCALAR_SCORE_COLUMNS or any(
        name == f"{base}_{suffix}"
        for base in SCALAR_SCORE_COLUMNS
        for suffix in CALIBRATED_SCORE_SUFFIXES
    ):
        return "feature_selection"
    if any(
        name == f"{prefix}_{suffix}"
        for prefix in ("fsl", "prior", "screening")
        for suffix in SELECTION_ALIAS_SUFFIXES
    ):
        return "feature_selection"
    if re.fullmatch(
        r"prior_screening_fusion_w\d{3}_logit(?:_(?:"
        + "|".join(CALIBRATED_SCORE_SUFFIXES)
        + r"))?",
        name,
    ) or re.fullmatch(
        r"fusion_w\d{3}_(?:" + "|".join(SELECTION_ALIAS_SUFFIXES) + r")",
        name,
    ):
        return "feature_selection"
    return ""


def _rank01_average_ties(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros(arr.shape[0], dtype=np.float64)
    valid = np.isfinite(arr)
    idx = np.flatnonzero(valid)
    if idx.size <= 1:
        return out
    vals = arr[idx]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(idx.size, dtype=np.float64)
    sorted_vals = vals[order]
    start = 0
    while start < sorted_vals.size:
        stop = start + 1
        while stop < sorted_vals.size and sorted_vals[stop] == sorted_vals[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * float(start + stop - 1)
        start = stop
    out[idx] = ranks / float(max(1, idx.size - 1))
    return out


def _rank01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(arr)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out
    if idx.size == 1:
        out[idx] = 0.0
        return out
    order = np.argsort(arr[idx], kind="mergesort")
    ranks = np.empty(idx.size, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, num=idx.size, dtype=np.float64)
    out[idx] = ranks
    return out


def _softmax_temperature_scores(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros(arr.shape[0], dtype=np.float64)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return out
    sub = arr[valid]
    mu = float(np.mean(sub))
    sd = float(np.std(sub))
    z = np.clip((sub - mu) / (sd + 1e-8), -30.0, 30.0)
    exp_z = np.exp(z - float(np.max(z)))
    denom = float(np.sum(exp_z))
    if denom > 0.0:
        out[np.flatnonzero(valid)] = exp_z / denom
    return out


def chunk_calibrate(values: np.ndarray, chunk_ids: np.ndarray | None, mode: str) -> np.ndarray:
    """Calibrate per-feature scores within their emission chunk.

    ``chunk_zscore`` de-means/scales inside each chunk (kills cross-chunk scale drift);
    ``chunk_rank01`` ranks to [0, 1] inside each chunk.  The additional modes mirror
    the S1 calibration-ablation harness: tie-aware ECDF, min/max, robust median/IQR,
    softmax-temperature, and a 50/50 global-rank plus chunk-zscore-rank blend.
    Sentinel/NaN entries are held out of the per-chunk statistics and filled to the
    chunk minimum afterwards so padded features always rank lowest.
    """
    mode = str(mode)
    if mode not in CALIBRATION_MODES:
        raise ValueError(f"unknown calibration mode {mode!r}; expected {CALIBRATION_MODES}")
    out = np.asarray(values, dtype=np.float64).copy()
    valid = np.isfinite(out) & (out > _PAD_SENTINEL + 1e-6)
    if mode == "none":
        return np.where(valid, out, np.nan)
    if chunk_ids is None:
        chunk_ids = np.zeros(out.shape[0], dtype=np.int64)
    chunk_ids = np.asarray(chunk_ids)
    result = np.full(out.shape[0], np.nan, dtype=np.float64)
    if mode == "blend":
        global_rank = _rank01(np.where(valid, out, np.nan))
        z_rank = _rank01(chunk_calibrate(out, chunk_ids, "chunk_zscore"))
        result = 0.5 * global_rank + 0.5 * z_rank
        fill = float(np.nanmin(result)) if np.any(np.isfinite(result)) else 0.0
        return np.where(np.isfinite(result), result, fill)

    for cid in np.unique(chunk_ids):
        m = (chunk_ids == cid) & valid
        if not np.any(m):
            continue
        x = out[m]
        if mode == "chunk_zscore":
            mu = float(np.mean(x))
            sd = float(np.std(x))
            result[m] = (x - mu) / sd if sd > 1e-12 else 0.0
        elif mode == "chunk_rank01":
            order = np.argsort(np.argsort(x))
            denom = max(1, x.shape[0] - 1)
            result[m] = order.astype(np.float64) / denom
        elif mode == "chunk_ecdf":
            result[m] = _rank01_average_ties(x)
        elif mode == "chunk_minmax":
            lo = float(np.min(x))
            hi = float(np.max(x))
            result[m] = (x - lo) / (hi - lo) if hi > lo else 0.5
        elif mode == "chunk_robust_iqr":
            center = float(np.median(x))
            q75, q25 = np.percentile(x, [75, 25])
            scale = float(q75 - q25)
            result[m] = (x - center) / (scale + 1e-8)
        elif mode == "chunk_softmax_temp":
            result[m] = _softmax_temperature_scores(x)
    # Fill held-out (padded/invalid) entries to the global min so they rank lowest.
    if np.any(~np.isfinite(result)):
        fill = float(np.nanmin(result)) if np.any(np.isfinite(result)) else 0.0
        result = np.where(np.isfinite(result), result, fill)
    return result


def _stack_object_column(series: Any, expected_dim: int | None = None) -> np.ndarray | None:
    """Stack an object/array parquet column into a dense [F, D] float array."""
    rows = []
    for v in series:
        try:
            a = _coerce_vector_cell(v)
        except Exception:
            return None
        rows.append(a)
    if not rows:
        return None
    dim = max(r.shape[0] for r in rows)
    if expected_dim is not None:
        dim = max(dim, expected_dim)
    mat = np.full((len(rows), dim), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        mat[i, : r.shape[0]] = r
    return mat


def _coerce_vector_cell(value: Any) -> np.ndarray:
    """Coerce a parquet vector cell into a 1-D float array.

    PyArrow-backed parquet writers usually preserve list/array cells, but some
    emitters serialize small vectors as JSON strings.  Accept both forms so the
    reader is not coupled to one storage backend.
    """
    raw = value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("empty vector cell")
        raw = json.loads(text)
    return np.asarray(raw, dtype=np.float64).ravel()


def _manifest_trust_record(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("diakrino_sidecar_trust_record", "sidecar_trust_record", "trust_record"):
        raw = payload.get(key)
        if isinstance(raw, dict):
            return normalize_diakrino_sidecar_trust_record(raw)
    return None


def _resolve_manifest_sidecar(path: Path, dataset_id: str | None) -> tuple[Path, dict[str, Any] | None] | None:
    """Resolve a dataset sidecar and trust record from a NAT-01 manifest JSON."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    base = path.parent
    wanted = None if dataset_id is None else str(dataset_id)
    trust_record = _manifest_trust_record(payload)
    manifest_sha = str(payload.get("checkpoint_sha256") or "")
    trust_sha = trust_record_checkpoint_sha256(trust_record)
    if trust_record is not None and manifest_sha and trust_sha and manifest_sha != trust_sha:
        return None

    sidecars = payload.get("sidecars")
    if isinstance(sidecars, list):
        for row in sidecars:
            if not isinstance(row, dict):
                continue
            row_dataset = row.get("dataset_id")
            if wanted is not None and str(row_dataset) != wanted:
                continue
            candidate = row.get("feature_logits_path") or row.get("path")
            if not candidate:
                continue
            p = Path(str(candidate))
            return (p if p.is_absolute() else (base / p)), trust_record

    datasets = payload.get("datasets")
    if isinstance(datasets, dict) and wanted is not None:
        row = datasets.get(wanted)
        if isinstance(row, str):
            p = Path(row)
            return (p if p.is_absolute() else (base / p)), trust_record
        if isinstance(row, dict):
            candidate = row.get("feature_logits_path") or row.get("path")
            if candidate:
                p = Path(str(candidate))
                return (p if p.is_absolute() else (base / p)), trust_record
    return None


class DiakrinoSidecar:
    """Per-dataset view over a persisted ``feature_logits`` sidecar parquet.

    Construct via :meth:`load`.  All vector accessors return arrays aligned to the
    dense original feature index ``0 .. n_features-1`` (``feature_index`` column),
    or ``None`` when the requested signal is unavailable.
    """

    def __init__(
        self,
        frame: Any,
        n_features: int,
        *,
        source_path: str = "",
        requested_path: str = "",
        dataset_id: str = "",
        trust_record: dict[str, Any] | None = None,
        allow_trust_override: bool = False,
        validated_identity: DiakrinoExpectedIdentity | None = None,
        required_head: str = "",
        artifact_identities: Mapping[str, Any] | None = None,
        qualification_record_sha256: str = "",
        source_emission_manifest_sha256: str = "",
        claim_manifest_sha256: str = "",
        inference_view_diagnostics: Mapping[str, Any] | None = None,
        inference_view_values: Mapping[str, Any] | None = None,
        native_null_payload: Mapping[str, Any] | None = None,
        paired_inference_view_payload: Mapping[str, Any] | None = None,
    ) -> None:
        self._df = frame
        self._n = int(n_features)
        self._source_path = str(source_path or "")
        self._requested_path = str(requested_path or "")
        self._dataset_id = str(dataset_id or "")
        self._trust_record = normalize_diakrino_sidecar_trust_record(trust_record)
        self._allow_trust_override = bool(allow_trust_override)
        self._validated_identity = validated_identity
        self._required_head = str(required_head or "")
        self._artifact_identities = dict(artifact_identities or {})
        self._qualification_record_sha256 = str(qualification_record_sha256 or "")
        self._source_emission_manifest_sha256 = str(source_emission_manifest_sha256 or "")
        self._claim_manifest_sha256 = str(claim_manifest_sha256 or "")
        self._inference_view_diagnostics = dict(inference_view_diagnostics or {})
        self._inference_view_values = dict(inference_view_values or {})
        self._native_null_payload = dict(native_null_payload or {})
        self._paired_inference_view_payload = dict(
            paired_inference_view_payload or {}
        )
        self._claim_eligible = bool(
            validated_identity is not None
            and self._trust_record is not None
            and self._required_head
            and len(self._qualification_record_sha256) == 64
            and len(self._source_emission_manifest_sha256) == 64
            and len(self._claim_manifest_sha256) == 64
            and trust_record_head_allowed(self._trust_record, self._required_head)
        )

    # -- construction --------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str | Path,
        dataset_id: str | None = None,
        *,
        allow_trust_override: bool = False,
    ) -> "DiakrinoSidecar | None":
        """Load a sidecar for diagnostics/backwards compatibility.

        ``path`` may point at a ``feature_logits`` parquet file directly, or at a
        directory containing ``feature_logits/<dataset_id>.parquet`` (or
        ``<dataset_id>.parquet``).  Returns ``None`` on any failure — missing pandas,
        missing file, unreadable parquet, empty frame — so callers degrade gracefully.

        This permissive API is never claim-bearing, including when it happens to
        resolve a v2 manifest.  Canonical validation must call :meth:`load_strict`
        with an independently reconstructed :class:`DiakrinoExpectedIdentity`.
        """
        pd = _import_pandas()
        if pd is None:
            return None
        try:
            p = Path(path)
            trust_record = None
            if p.is_file() and p.suffix.lower() == ".json":
                hit = _resolve_manifest_sidecar(p, dataset_id)
                if hit is None:
                    return None
                p, trust_record = hit
            if p.is_dir():
                cands = []
                if dataset_id:
                    cands = [
                        p / "feature_logits" / f"{dataset_id}.parquet",
                        p / f"{dataset_id}.parquet",
                        p / "manifest.json",
                    ]
                hit = next((c for c in cands if c.exists()), None)
                if hit is None:
                    return None
                if hit.suffix.lower() == ".json":
                    resolved = _resolve_manifest_sidecar(hit, dataset_id)
                    if resolved is None:
                        return None
                    hit, trust_record = resolved
                p = hit
            if not p.exists():
                return None
            if trust_record is not None and not bool(allow_trust_override):
                required = trust_record_required_calibration(trust_record)
                if required not in CALIBRATION_MODES or required == "none":
                    return None
                if not trust_record_discrete_filter_ok(trust_record):
                    return None
            df = pd.read_parquet(p)
            if df is None or len(df) == 0:
                return None
            if dataset_id is not None and "dataset_id" in df.columns:
                df = df[df["dataset_id"].astype(str) == str(dataset_id)]
                if len(df) == 0:
                    return None
            if "feature_index" in df.columns:
                df = df.sort_values("feature_index").reset_index(drop=True)
                n_features = int(df["feature_index"].max()) + 1
            else:
                df = df.reset_index(drop=True)
                n_features = len(df)
            return cls(
                df,
                n_features,
                source_path=str(p),
                requested_path=str(path),
                dataset_id=str(dataset_id or ""),
                trust_record=trust_record,
                allow_trust_override=bool(allow_trust_override),
            )
        except Exception:
            return None

    @classmethod
    def load_strict(
        cls,
        manifest_path: str | Path,
        expected_identity: DiakrinoExpectedIdentity | Mapping[str, Any],
        *,
        required_head: str = "feature_selection",
    ) -> "DiakrinoSidecar":
        """Load a claim-bearing v2 sidecar or raise on any identity mismatch.

        The manifest, per-dataset identity, exact support/query arrays, all three
        artifact byte identities, every artifact frame's row binding, feature-index
        coverage/order, and per-head trust are validated before a sidecar is returned.
        Direct parquet and legacy manifests are rejected.
        """

        pd = _import_pandas()
        if pd is None:
            raise DiakrinoSidecarIdentityError("pandas with parquet support is required")
        expected = (
            expected_identity
            if isinstance(expected_identity, DiakrinoExpectedIdentity)
            else DiakrinoExpectedIdentity.from_dict(expected_identity)
        )
        expected._validate_dimensions()
        claim_path = Path(manifest_path)
        if claim_path.suffix.lower() != ".json" or not claim_path.is_file():
            raise DiakrinoSidecarIdentityError(
                "strict DIAKRINO loading requires a v2 manifest JSON; direct parquet/directory paths are diagnostic-only"
            )
        try:
            claim_bytes = claim_path.read_bytes()
            claim_payload = json.loads(claim_bytes.decode("utf-8"))
        except Exception as exc:
            raise DiakrinoSidecarIdentityError("DIAKRINO claim manifest is not readable JSON") from exc
        if not isinstance(claim_payload, dict):
            raise DiakrinoSidecarIdentityError("DIAKRINO claim manifest must contain a JSON object")
        claim_manifest_sha256 = sha256_bytes(claim_bytes)
        manifest, row, observed = resolve_manifest_identity_record(
            manifest_path,
            dataset_id=expected.dataset_id,
            manifest_payload=claim_payload,
        )
        if observed.to_dict() != expected.to_dict():
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO manifest identity does not match the independently expected identity"
            )
        if expected.calibration_mode != DIAKRINO_REQUIRED_CALIBRATION_MODE:
            raise DiakrinoSidecarIdentityError(
                f"claim-bearing DIAKRINO sidecars require {DIAKRINO_REQUIRED_CALIBRATION_MODE!r} calibration"
            )
        trust_record = _manifest_trust_record(manifest)
        if trust_record is None:
            raise DiakrinoSidecarIdentityError("DIAKRINO manifest is missing a valid trust record")
        if trust_record_checkpoint_sha256(trust_record) != expected.checkpoint_sha256:
            raise DiakrinoSidecarIdentityError("DIAKRINO trust record is bound to another checkpoint")
        if trust_record_required_calibration(trust_record) != expected.calibration_mode:
            raise DiakrinoSidecarIdentityError("DIAKRINO trust record calibration identity mismatch")
        if not trust_record_discrete_filter_ok(trust_record):
            raise DiakrinoSidecarIdentityError("DIAKRINO trust record lacks the discrete-family safeguard")
        requested_head = str(required_head or "").strip()
        if not requested_head:
            raise DiakrinoSidecarIdentityError("strict loading requires a requested consumer head")
        if not trust_record_head_allowed(trust_record, requested_head):
            raise DiakrinoSidecarIdentityError(
                f"DIAKRINO trust record does not allow requested head {requested_head!r}"
            )

        qualification_ref = manifest.get("qualification_record_artifact")
        if not isinstance(qualification_ref, Mapping):
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO manifest is missing an immutable qualification-record artifact"
            )
        qualification_bytes, _, qualification_artifact = read_verified_artifact_reference(
            manifest_path,
            qualification_ref,
            label="qualification_record",
            require_relative_path=True,
        )
        trust_qualification_sha = str(
            trust_record.get("qualification_record_sha256") or ""
        ).strip().lower()
        if qualification_artifact.sha256 != trust_qualification_sha:
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO trust and qualification artifact SHA-256 identities disagree"
            )
        try:
            qualification_record = json.loads(qualification_bytes.decode("utf-8"))
        except Exception as exc:
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO qualification artifact is not readable JSON"
            ) from exc
        if not isinstance(qualification_record, dict):
            raise DiakrinoSidecarIdentityError("DIAKRINO qualification artifact must be a JSON object")
        checkpoint = qualification_record.get("checkpoint")
        qualification_checkpoint = (
            str(checkpoint.get("sha256") or "").strip().lower()
            if isinstance(checkpoint, Mapping)
            else ""
        )
        if qualification_checkpoint != expected.checkpoint_sha256:
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO qualification artifact is bound to another checkpoint"
            )
        source_emission_ref = manifest.get("source_emission_manifest_artifact")
        if not isinstance(source_emission_ref, Mapping):
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO claim manifest is missing its immutable source-emission manifest"
            )
        source_emission_bytes, _, source_emission_artifact = read_verified_artifact_reference(
            manifest_path,
            source_emission_ref,
            label="source_emission_manifest",
            require_relative_path=True,
        )
        try:
            source_emission = json.loads(source_emission_bytes.decode("utf-8"))
        except Exception as exc:
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO source-emission artifact is not readable JSON"
            ) from exc
        if not isinstance(source_emission, dict) or str(
            source_emission.get("schema_version") or ""
        ) != "diakrino_sidecar_manifest_v2":
            raise DiakrinoSidecarIdentityError("DIAKRINO source-emission manifest is not v2")
        source_phase = source_emission.get("binding_phase")
        if source_phase not in (None, "", "emission_manifest"):
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO source-emission artifact is not an emission-phase manifest"
            )
        if source_emission.get("source_emission_manifest_artifact") is not None:
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO source-emission artifact must not reference another claim source"
            )
        if trust_record.get("claim_scope") != "evaluation":
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO strict claim is not authorized for evaluation consumption"
            )
        if str(trust_record.get("source_manifest_sha256") or "").strip().lower() != (
            source_emission_artifact.sha256
        ):
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO trust and source-emission manifest SHA-256 identities disagree"
            )
        source_trust = source_emission.get("diakrino_sidecar_trust_record")
        if not isinstance(source_trust, Mapping) or trust_record_head_allowed(
            source_trust, "feature_selection"
        ):
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO source-emission artifact must explicitly deny feature-selection trust"
            )
        source_rows = source_emission.get("sidecars")
        source_matches = [
            item
            for item in list(source_rows or [])
            if isinstance(item, Mapping)
            and str(item.get("dataset_id") or "") == expected.dataset_id
        ]
        if len(source_matches) != 1:
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO source-emission manifest lacks the exact dataset record"
            )
        source_row = source_matches[0]
        if source_row.get("identity") != expected.to_dict():
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO source-emission and claim identity bindings disagree"
            )
        source_artifacts = source_row.get("artifacts")
        claim_artifacts = row.get("artifacts")
        if not isinstance(source_artifacts, Mapping) or not isinstance(claim_artifacts, Mapping):
            raise DiakrinoSidecarIdentityError("DIAKRINO source/claim artifact identities are malformed")
        required_artifacts = ("feature_logits", "query_class_logits", "aux_logits")
        optional_artifacts = (
            "feature_embeddings",
            "inference_views",
            "native_nulls",
            "paired_inference_views",
        )
        for artifact_name in (
            *required_artifacts,
            *(
                name
                for name in optional_artifacts
                if name in source_artifacts or name in claim_artifacts
            ),
        ):
            source_artifact = source_artifacts.get(artifact_name)
            claim_artifact = claim_artifacts.get(artifact_name)
            if not isinstance(source_artifact, Mapping) or not isinstance(claim_artifact, Mapping):
                raise DiakrinoSidecarIdentityError("DIAKRINO source/claim artifact is missing")
            if (
                source_artifact.get("sha256") != claim_artifact.get("sha256")
                or source_artifact.get("size_bytes") != claim_artifact.get("size_bytes")
                or source_artifact.get("path") != claim_artifact.get("path")
            ):
                raise DiakrinoSidecarIdentityError(
                    f"DIAKRINO source/claim {artifact_name} byte identities disagree"
                )
        if requested_head == "feature_selection":
            qualification_covers_identity = source_emission_qualification_covers_identity(
                qualification_record,
                expected,
                source_manifest_sha256=source_emission_artifact.sha256,
            )
        else:
            qualification_covers_identity = qualification_record_covers_identity(
                qualification_record,
                expected,
                consumer_class=requested_head,
                source_manifest_sha256=source_emission_artifact.sha256,
            )
        if not qualification_covers_identity:
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO qualification artifact does not cover the exact requested identity/head"
            )

        artifact_bytes, artifact_paths = read_verified_artifacts(
            manifest_path,
            row,
            require_relative_paths=True,
        )
        try:
            feature_frame = pd.read_parquet(io.BytesIO(artifact_bytes["feature_logits"]))
            query_frame = pd.read_parquet(io.BytesIO(artifact_bytes["query_class_logits"]))
            aux_frame = pd.read_parquet(io.BytesIO(artifact_bytes["aux_logits"]))
        except Exception as exc:
            raise DiakrinoSidecarIdentityError("DIAKRINO artifact bytes are not readable parquet") from exc

        for label, frame in (
            ("feature_logits", feature_frame),
            ("query_class_logits", query_frame),
            ("aux_logits", aux_frame),
        ):
            validate_frame_identity(frame, expected, artifact=label)
            if "dataset_id" not in frame.columns:
                raise DiakrinoSidecarIdentityError(f"DIAKRINO {label} frame has no dataset_id column")
            if set(frame["dataset_id"].astype(str).tolist()) != {expected.dataset_id}:
                raise DiakrinoSidecarIdentityError(f"DIAKRINO {label} dataset identity mismatch")

        n_features = int(expected.feature_order.count)
        for column in ("feature_index", "chunk_id"):
            if column not in feature_frame.columns:
                raise DiakrinoSidecarIdentityError(
                    f"DIAKRINO feature_logits frame has no {column!r} column"
                )
        if len(feature_frame) != n_features:
            raise DiakrinoSidecarIdentityError(
                f"DIAKRINO feature row count mismatch: expected={n_features} observed={len(feature_frame)}"
            )
        try:
            feature_indices = np.asarray(feature_frame["feature_index"].to_numpy())
            if not np.issubdtype(feature_indices.dtype, np.integer):
                raise TypeError("not integer")
            feature_indices = feature_indices.astype(np.int64, copy=False)
        except Exception as exc:
            raise DiakrinoSidecarIdentityError("DIAKRINO feature indices are not exact integers") from exc
        if not np.array_equal(feature_indices, np.arange(n_features, dtype=np.int64)):
            raise DiakrinoSidecarIdentityError(
                "DIAKRINO feature indices are negative, duplicated, gapped, or reordered"
            )
        uniform_columns = {
            DIAKRINO_UNIFORM_VIEW_RANK_COLUMN,
            "diakrino_uniform_view_rank_std",
            "diakrino_uniform_view_class_count",
            "diakrino_uniform_view_score_source",
            "diakrino_uniform_view_ids_json",
            "diakrino_inference_view_schema_version",
            "diakrino_inference_view_id",
            "diakrino_inference_view_seed",
            "diakrino_inference_view_record_sha256",
        }
        view_enabled = "inference_views" in artifact_bytes or bool(
            uniform_columns.intersection(feature_frame.columns)
        )
        view_diagnostics: dict[str, Any] = {}
        view_values: dict[str, Any] = {}
        if view_enabled:
            if expected.sidecar_schema_version != 2:
                raise DiakrinoSidecarIdentityError(
                    "uniform-view DIAKRINO scores require sidecar schema v2"
                )
            if "inference_views" not in artifact_bytes or not uniform_columns.issubset(
                feature_frame.columns
            ):
                raise DiakrinoSidecarIdentityError(
                    "uniform-view DIAKRINO scores require the complete verified view contract"
                )
            try:
                raw_class_counts = np.asarray(
                    feature_frame["diakrino_uniform_view_class_count"].to_numpy()
                )
                if np.issubdtype(raw_class_counts.dtype, np.bool_) or not np.issubdtype(
                    raw_class_counts.dtype, np.integer
                ):
                    raise DiakrinoViewError("uniform-view class counts are not exact integers")
                class_counts = exact_int64_vector(
                    raw_class_counts,
                    label="uniform-view class counts",
                )
                if (
                    class_counts.size != n_features
                    or np.any(class_counts < 2)
                    or np.unique(class_counts).size != 1
                ):
                    raise DiakrinoViewError("uniform-view class-count identity is invalid")
                view_payload = json.loads(artifact_bytes["inference_views"].decode("utf-8"))
                validated_view = validate_view_artifact(
                    view_payload,
                    binding_sha256=expected.binding_sha256,
                    n_features=n_features,
                    n_support=len(row["support_indices"]),
                    n_classes=int(class_counts[0]),
                )
                identity_view = validated_view.views[0]
                identity_view_record_sha256 = canonical_json_sha256(
                    identity_view.manifest_record(),
                    payload_schema_version="diakrino_inference_view_record_v1",
                )
                required_frame_view_columns = {
                    "diakrino_inference_view_schema_version",
                    "diakrino_inference_view_id",
                    "diakrino_inference_view_seed",
                    "diakrino_inference_view_record_sha256",
                }
                for artifact_name, frame in (
                    ("feature_logits", feature_frame),
                    ("query_class_logits", query_frame),
                    ("aux_logits", aux_frame),
                ):
                    if not required_frame_view_columns.issubset(frame.columns):
                        raise DiakrinoViewError(
                            f"{artifact_name} lacks its frozen identity-view binding"
                        )
                    if set(
                        frame["diakrino_inference_view_schema_version"].astype(str).tolist()
                    ) != {"diakrino_inference_view_record_v1"} or set(
                        frame["diakrino_inference_view_id"].astype(str).tolist()
                    ) != {identity_view.view_id} or set(
                        frame["diakrino_inference_view_record_sha256"].astype(str).tolist()
                    ) != {identity_view_record_sha256}:
                        raise DiakrinoViewError(
                            f"{artifact_name} has a cross-wired identity-view binding"
                        )
                    frame_seeds = {
                        exact_int64_scalar(
                            value,
                            label=f"{artifact_name} identity-view seed",
                            minimum=0,
                        )
                        for value in frame["diakrino_inference_view_seed"].tolist()
                    }
                    if frame_seeds != {int(identity_view.seed)}:
                        raise DiakrinoViewError(
                            f"{artifact_name} has a cross-wired identity-view seed"
                        )
                observed_numeric: dict[str, np.ndarray] = {}
                for column in (
                    DIAKRINO_UNIFORM_VIEW_RANK_COLUMN,
                    "diakrino_uniform_view_rank_std",
                ):
                    raw_values = np.asarray(feature_frame[column].to_numpy())
                    if np.issubdtype(raw_values.dtype, np.bool_) or not np.issubdtype(
                        raw_values.dtype, np.number
                    ):
                        raise DiakrinoViewError(f"{column} is not exactly numeric")
                    values = raw_values.astype(np.float64, copy=False)
                    if values.shape != (n_features,) or not np.all(np.isfinite(values)):
                        raise DiakrinoViewError(f"{column} is incomplete or non-finite")
                    observed_numeric[column] = values
                if not (
                    np.array_equal(
                        np.asarray(validated_view.uniform_rank, dtype=np.float64),
                        observed_numeric[DIAKRINO_UNIFORM_VIEW_RANK_COLUMN],
                    )
                    and np.array_equal(
                        np.asarray(validated_view.uniform_rank_std, dtype=np.float64),
                        observed_numeric["diakrino_uniform_view_rank_std"],
                    )
                ):
                    raise DiakrinoViewError("uniform-view columns differ from the sealed aggregate")
                if set(
                    feature_frame["diakrino_uniform_view_score_source"].astype(str).tolist()
                ) != {DIAKRINO_VIEW_SCORE_SOURCE}:
                    raise DiakrinoViewError("uniform-view score-source identity is invalid")
                declared_view_ids = set(
                    feature_frame["diakrino_uniform_view_ids_json"].astype(str).tolist()
                )
                if len(declared_view_ids) != 1:
                    raise DiakrinoViewError("uniform-view id identity is inconsistent")
                parsed_view_ids = json.loads(next(iter(declared_view_ids)))
                if not isinstance(parsed_view_ids, list) or tuple(parsed_view_ids) != (
                    DIAKRINO_FROZEN_VIEW_IDS
                ):
                    raise DiakrinoViewError("uniform-view id identity is invalid")
                view_diagnostics = validated_view.diagnostics()
                view_values = {
                    "uniform_rank": list(validated_view.uniform_rank),
                    "uniform_rank_std": list(validated_view.uniform_rank_std),
                    "view_ids": list(validated_view.view_ids),
                    "rank01_by_view": [
                        list(row) for row in validated_view.rank01_by_view
                    ],
                }
            except DiakrinoSidecarIdentityError:
                raise
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                DiakrinoViewError,
                KeyError,
                TypeError,
                ValueError,
                OverflowError,
            ) as exc:
                raise DiakrinoSidecarIdentityError(
                    "uniform-view DIAKRINO artifact fails the frozen view contract"
                ) from exc
        if bool(feature_frame["chunk_id"].isna().any()):
            raise DiakrinoSidecarIdentityError("DIAKRINO feature chunk ids are incomplete")
        try:
            feature_chunks = exact_int64_vector(
                feature_frame["chunk_id"].to_numpy(),
                label="feature chunk ids",
            )
        except Exception as exc:
            raise DiakrinoSidecarIdentityError("DIAKRINO feature chunk ids are malformed") from exc
        chunk_ids = np.unique(feature_chunks)
        if not np.array_equal(chunk_ids, np.arange(chunk_ids.size, dtype=np.int64)):
            raise DiakrinoSidecarIdentityError("DIAKRINO feature chunk ids are not dense from zero")

        for column in ("query_row_index", "chunk_id", "query_split"):
            if column not in query_frame.columns:
                raise DiakrinoSidecarIdentityError(
                    f"DIAKRINO query_class_logits frame has no {column!r} column"
                )
        if query_frame.empty:
            raise DiakrinoSidecarIdentityError("DIAKRINO query_class_logits frame is empty")
        query_splits = query_frame["query_split"].astype(str).str.strip().str.lower()
        if not bool(query_splits.isin(("query", "test", "heldout", "held_out", "holdout")).all()):
            raise DiakrinoSidecarIdentityError("DIAKRINO query frame contains non-held-out rows")
        manifest_query = exact_int64_vector(
            row["query_indices"], label="manifest query indices"
        )
        query_chunk_values = exact_int64_vector(
            query_frame["chunk_id"].to_numpy(), label="query chunk ids"
        )
        observed_query_chunks = list(dict.fromkeys(query_chunk_values.tolist()))
        if observed_query_chunks != [int(value) for value in chunk_ids.tolist()]:
            raise DiakrinoSidecarIdentityError("DIAKRINO query chunks do not match feature chunks")
        for chunk_id in observed_query_chunks:
            chunk_rows = query_frame.iloc[np.flatnonzero(query_chunk_values == chunk_id)]
            try:
                row_indices = exact_int64_vector(
                    chunk_rows["query_row_index"].to_numpy(),
                    label="query row indices",
                )
            except Exception as exc:
                raise DiakrinoSidecarIdentityError("DIAKRINO query row indices are malformed") from exc
            if not np.array_equal(row_indices, manifest_query):
                raise DiakrinoSidecarIdentityError(
                    f"DIAKRINO query rows for chunk {chunk_id} do not exactly match the ordered split"
                )

        if "chunk_id" not in aux_frame.columns or aux_frame.empty:
            raise DiakrinoSidecarIdentityError("DIAKRINO aux_logits frame has no chunk identity")
        try:
            aux_chunks = exact_int64_vector(
                aux_frame["chunk_id"].to_numpy(), label="aux chunk ids"
            )
        except Exception as exc:
            raise DiakrinoSidecarIdentityError("DIAKRINO aux chunk ids are malformed") from exc
        if not np.array_equal(aux_chunks, chunk_ids):
            raise DiakrinoSidecarIdentityError("DIAKRINO aux chunks do not match feature chunks")

        artifacts = row.get("artifacts")
        assert isinstance(artifacts, Mapping)
        native_null_payload: dict[str, Any] = {}
        paired_inference_view_payload: dict[str, Any] = {}
        if "native_nulls" in artifact_bytes:
            try:
                raw_native_nulls = json.loads(
                    artifact_bytes["native_nulls"].decode("utf-8")
                )
            except Exception as exc:
                raise DiakrinoSidecarIdentityError(
                    "DIAKRINO native-null artifact is not readable JSON"
                ) from exc
            if not isinstance(raw_native_nulls, dict):
                raise DiakrinoSidecarIdentityError(
                    "DIAKRINO native-null artifact must be a JSON object"
                )
            native_null_payload = raw_native_nulls
        if "paired_inference_views" in artifact_bytes:
            try:
                raw_paired_views = json.loads(
                    artifact_bytes["paired_inference_views"].decode("utf-8")
                )
            except Exception as exc:
                raise DiakrinoSidecarIdentityError(
                    "DIAKRINO paired inference-view artifact is not readable JSON"
                ) from exc
            if not isinstance(raw_paired_views, dict):
                raise DiakrinoSidecarIdentityError(
                    "DIAKRINO paired inference-view artifact must be a JSON object"
                )
            paired_inference_view_payload = raw_paired_views
        return cls(
            feature_frame.reset_index(drop=True),
            n_features,
            source_path=artifact_paths["feature_logits"],
            requested_path=str(manifest_path),
            dataset_id=expected.dataset_id,
            trust_record=trust_record,
            allow_trust_override=False,
            validated_identity=expected,
            required_head=requested_head,
            artifact_identities=dict(artifacts),
            qualification_record_sha256=qualification_artifact.sha256,
            source_emission_manifest_sha256=source_emission_artifact.sha256,
            claim_manifest_sha256=claim_manifest_sha256,
            inference_view_diagnostics=view_diagnostics,
            inference_view_values=view_values,
            native_null_payload=native_null_payload,
            paired_inference_view_payload=paired_inference_view_payload,
        )

    # -- alignment helpers ---------------------------------------------------
    @property
    def n_features(self) -> int:
        return self._n

    def resolution_diagnostics(self) -> dict[str, Any]:
        """JSON-safe metadata describing the sidecar artifact that was loaded."""
        return {
            "loaded": True,
            "requested_path": self._requested_path,
            "source_path": self._source_path,
            "dataset_id": self._dataset_id,
            "n_features": int(self._n),
            "row_count": int(len(self._df)),
            "has_feature_index": bool("feature_index" in self._df.columns),
            "has_chunk_id": bool("chunk_id" in self._df.columns),
            "trust_record_present": bool(self._trust_record is not None),
            "trust_checkpoint_sha256": trust_record_checkpoint_sha256(self._trust_record),
            "required_calibration_mode": trust_record_required_calibration(self._trust_record)
            if self._trust_record is not None
            else "",
            "trust_override": bool(self._allow_trust_override),
            "identity_mode": "strict" if self._validated_identity is not None else "diagnostic_legacy",
            "claim_eligible": bool(self._claim_eligible),
            "validated_binding_sha256": (
                self._validated_identity.binding_sha256
                if self._validated_identity is not None
                else ""
            ),
            "required_head": self._required_head,
            "artifact_identities": dict(self._artifact_identities),
            "qualification_record_sha256": self._qualification_record_sha256,
            "source_emission_manifest_sha256": self._source_emission_manifest_sha256,
            "claim_manifest_sha256": self._claim_manifest_sha256,
            "inference_view_diagnostics": dict(self._inference_view_diagnostics),
        }

    @property
    def claim_eligible(self) -> bool:
        """Whether strict identity and requested-head trust both passed."""

        return bool(self._claim_eligible)

    @property
    def binding_sha256(self) -> str:
        """Validated identity binding digest, empty for permissive loads."""

        return (
            self._validated_identity.binding_sha256
            if self._validated_identity is not None
            else ""
        )

    def validated_identity(self) -> dict[str, Any] | None:
        """Return the strict JSON-native identity, or ``None`` for diagnostics."""

        return (
            None
            if self._validated_identity is None
            else self._validated_identity.to_dict()
        )

    def trust_record(self) -> dict[str, Any] | None:
        """Return the JSON-native sidecar trust record, when loaded from a manifest."""

        return None if self._trust_record is None else dict(self._trust_record)

    def native_null_payload(self) -> dict[str, Any] | None:
        """Return verified bytes parsed from the optional native-null artifact.

        Semantic validation still requires canonical support X/y and the exact
        inference-view artifact, so canonical runners must call
        ``validate_native_null_artifact`` before consuming these values.
        """

        if not self._native_null_payload:
            return None
        return json.loads(json.dumps(self._native_null_payload))

    def paired_inference_view_payload(self) -> dict[str, Any] | None:
        """Return verified bytes of the optional paired P2 view ledger."""

        if not self._paired_inference_view_payload:
            return None
        return json.loads(json.dumps(self._paired_inference_view_payload))

    def inference_view_values(self) -> dict[str, Any] | None:
        """Return values strictly rederived from the frozen view artifact."""

        if not self._inference_view_values:
            return None
        return json.loads(json.dumps(self._inference_view_values))

    def _resolve_calibration_mode(self, calibrate: str | None) -> str:
        mode = str(calibrate or "chunk_zscore").strip().lower()
        if mode in {"auto", "trust", "required"} and self._trust_record is not None:
            mode = trust_record_required_calibration(self._trust_record)
        resolved = mode or "chunk_zscore"
        if self._validated_identity is not None and not self._allow_trust_override:
            if self._trust_record is None:
                raise DiakrinoSidecarIdentityError(
                    "strict DIAKRINO sidecar is missing its calibration trust record"
                )
            required = trust_record_required_calibration(self._trust_record)
            if resolved != required:
                raise ValueError(
                    "claim-bearing DIAKRINO sidecars require their identity-bound "
                    f"calibration mode {required!r}; got {resolved!r}"
                )
        return resolved

    def _head_allowed(self, head: str) -> bool:
        if self._trust_record is None or self._allow_trust_override:
            return True
        return trust_record_head_allowed(self._trust_record, str(head))

    def _strict_accessor_head_allowed(self, head: str) -> bool:
        """Enforce per-signal trust on claim-bearing objects only.

        Diagnostic/legacy objects intentionally retain their non-claim-bearing
        accessor behavior.  A strict object always has a validated identity and
        must re-authorize the exact head at every learned-signal accessor instead
        of inheriting the head requested by :meth:`load_strict`.
        """

        if self._validated_identity is None:
            return True
        requested = str(head)
        return requested == self._required_head and self._head_allowed(requested)

    def _chunk_ids(self) -> np.ndarray | None:
        if "chunk_id" in self._df.columns:
            return np.asarray(self._df["chunk_id"].to_numpy())
        return None

    def _dense(self, local: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """Scatter a per-row vector onto the dense 0..n_features-1 feature axis."""
        out = np.full(self._n, fill, dtype=np.float64)
        if "feature_index" in self._df.columns:
            idx = np.asarray(self._df["feature_index"].to_numpy(), dtype=np.int64)
            ok = (idx >= 0) & (idx < self._n)
            out[idx[ok]] = np.asarray(local, dtype=np.float64)[ok]
        else:
            m = min(self._n, local.shape[0])
            out[:m] = np.asarray(local, dtype=np.float64)[:m]
        return out

    def _dense_matrix(self, local: np.ndarray, *, fill: float = np.nan) -> np.ndarray:
        """Scatter per-row matrix values onto dense original feature indices."""
        local = np.asarray(local, dtype=np.float64)
        out = np.full((self._n, local.shape[1]), fill, dtype=np.float64)
        if "feature_index" in self._df.columns:
            idx = np.asarray(self._df["feature_index"].to_numpy(), dtype=np.int64)
            ok = (idx >= 0) & (idx < self._n)
            out[idx[ok]] = local[ok]
        else:
            m = min(self._n, local.shape[0])
            out[:m] = local[:m]
        return out

    def _apply_discrete_skip_to_vector(self, values: np.ndarray) -> np.ndarray:
        mask = self.discrete_skip_mask()
        if mask is None:
            return values
        out = np.asarray(values, dtype=np.float64).copy()
        if out.shape[0] == mask.shape[0]:
            out[np.asarray(mask, dtype=bool)] = np.nan
        return out

    def _apply_discrete_skip_to_matrix(self, values: np.ndarray) -> np.ndarray:
        mask = self.discrete_skip_mask()
        if mask is None:
            return values
        out = np.asarray(values, dtype=np.float64).copy()
        if out.shape[0] == mask.shape[0]:
            out[np.asarray(mask, dtype=bool), :] = np.nan
        return out

    # -- scalar score surfaces (selection phase) -----------------------------
    def scalar_scores(
        self,
        column: str = "prior_logit",
        *,
        calibrate: str = "chunk_zscore",
        require_calibration_safe: bool = False,
    ) -> np.ndarray | None:
        """Dense chunk-calibrated per-feature score vector for ``column``.

        Aligned to ``0..n_features-1``.  ``require_calibration_safe`` rejects
        chunk-contaminated columns (anything outside ``CALIBRATION_SAFE_COLUMNS``)
        when used with ``calibrate='none'``.
        """
        column = str(column)
        if self._validated_identity is not None:
            trust_head = _strict_scalar_column_trust_head(column)
            if not trust_head or not self._strict_accessor_head_allowed(trust_head):
                return None
        if column == DIAKRINO_UNIFORM_VIEW_RANK_COLUMN:
            if str(calibrate or "none").strip().lower() != "none":
                raise ValueError(
                    "uniform-view DIAKRINO ranks are already frozen calibrated aggregates; "
                    "use calibrate='none'"
                )
            calibrate = "none"
        else:
            calibrate = self._resolve_calibration_mode(calibrate)
        if column not in self._df.columns:
            return None
        if (
            self._trust_record is not None
            and not self._allow_trust_override
            and calibrate == "none"
            and column not in CALIBRATION_SAFE_COLUMNS
        ):
            raise ValueError(
                f"{column!r} is protected by sidecar trust record; use "
                f"{trust_record_required_calibration(self._trust_record)!r} or load with allow_trust_override=True"
            )
        if (
            require_calibration_safe
            and calibrate == "none"
            and column not in CALIBRATION_SAFE_COLUMNS
        ):
            raise ValueError(
                f"{column!r} is chunk-contaminated; use one of {CALIBRATION_MODES[1:]} "
                f"or one of {sorted(CALIBRATION_SAFE_COLUMNS)}"
            )
        raw = np.asarray(self._df[column].to_numpy(), dtype=np.float64)
        cal = chunk_calibrate(raw, self._chunk_ids(), calibrate)
        return self._dense(cal, fill=float(np.nanmin(cal)) if np.any(np.isfinite(cal)) else 0.0)

    # -- schema-v2 optional surfaces -----------------------------------------
    def schema_v2_scalar_scores(self, column: str, *, calibrate: str = "chunk_zscore") -> np.ndarray | None:
        """Dense scalar schema-v2 score with explicit calibration-safety checks.

        Schema-v2 scalar outputs are conservative by default: they are treated as
        chunk-contaminated unless explicitly whitelisted, so raw global use is
        rejected.  The discrete/nuisance family skip mask is applied after dense
        alignment when the family head is present.
        """
        column = str(column)
        calibrate = self._resolve_calibration_mode(calibrate)
        if column not in SCHEMA_V2_SCALAR_COLUMNS:
            raise ValueError(f"unknown schema-v2 scalar column {column!r}")
        if not self._strict_accessor_head_allowed(
            SCHEMA_V2_SCALAR_TRUST_HEADS[column]
        ):
            return None
        if column not in self._df.columns:
            return None
        if calibrate == "none" and column in SCHEMA_V2_CHUNK_CONTAMINATED_COLUMNS:
            raise ValueError(
                f"{column!r} is chunk-contaminated; use one of {CALIBRATION_MODES[1:]} "
                "for cross-chunk use"
            )
        raw = np.asarray(self._df[column].to_numpy(), dtype=np.float64)
        cal = chunk_calibrate(raw, self._chunk_ids(), calibrate)
        dense = self._dense(cal, fill=float(np.nanmin(cal)) if np.any(np.isfinite(cal)) else 0.0)
        return self._apply_discrete_skip_to_vector(dense)

    @staticmethod
    def _reconstruction_prefix(kind: str) -> str:
        key = str(kind or "population").strip().lower()
        if key in {"population", "pop"}:
            return "population_reconstruction"
        if key in {"population_class", "class", "pop_class"}:
            return "population_class_reconstruction"
        raise ValueError("kind must be 'population' or 'population_class'")

    def reconstruction_vectors(self, kind: str = "population", value: str = "residuals") -> np.ndarray | None:
        """Dense schema-v2 reconstruction vector matrix for predictions/targets/residuals."""
        prefix = self._reconstruction_prefix(kind)
        if not self._strict_accessor_head_allowed(RECONSTRUCTION_TRUST_HEADS[prefix]):
            return None
        suffix = str(value or "residuals").strip().lower()
        if suffix not in {"predictions", "targets", "residuals"}:
            raise ValueError("value must be predictions, targets, or residuals")
        column = f"{prefix}_{suffix}"
        if column not in self._df.columns:
            return None
        mat = _stack_object_column(self._df[column])
        if mat is None:
            return None
        return self._apply_discrete_skip_to_matrix(self._dense_matrix(mat))

    def reconstruction_mse(self, kind: str = "population", *, calibrate: str = "chunk_zscore") -> np.ndarray | None:
        prefix = self._reconstruction_prefix(kind)
        return self.schema_v2_scalar_scores(f"{prefix}_mse", calibrate=calibrate)

    def conformal_scores(self, *, calibrate: str = "chunk_zscore") -> np.ndarray | None:
        return self.schema_v2_scalar_scores("conformal_score", calibrate=calibrate)

    def conformal_selection_probabilities(self, *, calibrate: str = "chunk_zscore") -> np.ndarray | None:
        return self.schema_v2_scalar_scores("conformal_selection_probability", calibrate=calibrate)

    def feature_embeddings(self) -> np.ndarray | None:
        """Diagnostic-only feature embedding matrix.

        Strict loading does not yet seal the optional external NPZ bytes, so no
        claim-bearing object may expose embeddings even if a future trust record
        enables the head.  Permissive loaders retain their diagnostic behavior.
        """
        if not self._strict_accessor_head_allowed("feature_embeddings"):
            return None
        if self._validated_identity is not None:
            return None
        if "feature_embedding" in self._df.columns:
            mat = _stack_object_column(self._df["feature_embedding"])
            return None if mat is None else self._apply_discrete_skip_to_matrix(self._dense_matrix(mat))
        if "feature_embeddings_path" not in self._df.columns or "feature_embedding_row" not in self._df.columns:
            return None
        raw_path = ""
        for value in self._df["feature_embeddings_path"].to_numpy():
            text = str(value or "").strip()
            if text:
                raw_path = text
                break
        if not raw_path:
            return None
        p = Path(raw_path)
        if not p.is_absolute():
            base = Path(self._source_path).parent if self._source_path else Path(".")
            if base.name == "feature_logits":
                base = base.parent
            p = base / p
        try:
            with np.load(p, allow_pickle=False) as payload:
                embeddings = np.asarray(payload["feature_embedding"], dtype=np.float64)
            rows = np.asarray(self._df["feature_embedding_row"].to_numpy(), dtype=np.int64)
            if embeddings.ndim != 2 or rows.size != len(self._df):
                return None
            ok = (rows >= 0) & (rows < embeddings.shape[0])
            local = np.full((len(self._df), embeddings.shape[1]), np.nan, dtype=np.float64)
            local[ok] = embeddings[rows[ok]]
        except Exception:
            return None
        return self._apply_discrete_skip_to_matrix(self._dense_matrix(local))

    # -- dataset-level selector routing head --------------------------------
    def _selector_weight_vector_from_columns(
        self,
        *,
        object_column: str,
        scalar_prefixes: tuple[str, ...],
    ) -> tuple[dict[str, float] | None, dict[str, Any]]:
        values: np.ndarray | None = None
        source = ""
        rows_examined = 0
        vector_lengths: list[int] = []
        if object_column in self._df.columns:
            for raw in self._df[object_column].to_numpy():
                rows_examined += 1
                try:
                    arr = _coerce_vector_cell(raw)
                except Exception:
                    continue
                vector_lengths.append(int(arr.size))
                if arr.size > 0 and np.any(np.isfinite(arr) & (arr > 0.0)):
                    values = arr
                    source = object_column
                    break

        scalar_columns: list[str] = []
        if values is None:
            vals = []
            found = False
            for name in DIAKRINO_SELECTOR_POOL:
                raw_val = None
                for prefix in scalar_prefixes:
                    col = f"{prefix}{name}"
                    if col in self._df.columns:
                        raw_val = self._df[col].iloc[0]
                        scalar_columns.append(col)
                        break
                if raw_val is None:
                    vals.append(np.nan)
                    continue
                try:
                    vals.append(float(raw_val))
                    found = True
                except Exception:
                    vals.append(np.nan)
            if found:
                values = np.asarray(vals, dtype=np.float64)
                source = "scalar_pool_columns"

        diagnostics: dict[str, Any] = {
            "source": source,
            "object_column": object_column,
            "object_rows_examined": int(rows_examined),
            "object_vector_lengths": sorted(set(vector_lengths)),
            "scalar_columns": scalar_columns,
            "selector_pool": list(DIAKRINO_SELECTOR_POOL),
        }
        if values is None:
            diagnostics["status"] = "missing"
            return None, diagnostics
        out: dict[str, float] = {}
        finite_count = 0
        for name, val in zip(DIAKRINO_SELECTOR_POOL, values.tolist()):
            try:
                f = float(val)
            except Exception:
                continue
            if np.isfinite(f):
                finite_count += 1
            if np.isfinite(f) and f >= 0.0:
                out[str(name)] = f
        total = float(sum(out.values()))
        diagnostics["finite_pool_member_count"] = int(finite_count)
        diagnostics["raw_positive_mass"] = float(total)
        if total <= 0.0:
            diagnostics["status"] = "zero_mass"
            return None, diagnostics
        normalized = {name: float(val / total) for name, val in out.items()}
        diagnostics["status"] = "usable"
        diagnostics["sum_to_one_abs_error"] = abs(float(sum(normalized.values())) - 1.0)
        diagnostics["normalized"] = dict(normalized)
        return normalized, diagnostics

    def selector_weights(self) -> dict[str, float] | None:
        """Dataset-level DIAKRINO selector weights keyed by experimental selector-pool name.

        The preferred sidecar representation is one object column named
        ``selector_weights`` containing a length-5 vector.  For forwards/backwards
        compatibility this accessor also accepts explicit scalar columns named
        ``selector_weight_<pool>`` or ``selector_weights_<pool>``.
        """
        if (
            not self._head_allowed("selector_weights")
            or not self._strict_accessor_head_allowed("selector_weights")
        ):
            return None
        weights, _diagnostics = self._selector_weight_vector_from_columns(
            object_column="selector_weights",
            scalar_prefixes=("selector_weight_", "selector_weights_"),
        )
        return weights

    def selector_weights_candidate(self) -> dict[str, float] | None:
        """Report-only predeclared dataset-level selector-weight candidate.

        This accessor deliberately does **not** feed the MNPO selector-prior
        consumer.  It reads only ``selector_weights_candidate`` columns so
        candidate emissions can be audited and qualified before any future
        promotion to the trusted ``selector_weights`` surface.
        """

        if not self._strict_accessor_head_allowed("selector_weights_candidate"):
            return None

        weights, _diagnostics = self._selector_weight_vector_from_columns(
            object_column="selector_weights_candidate",
            scalar_prefixes=("selector_weight_candidate_", "selector_weights_candidate_"),
        )
        return weights

    def selector_weights_candidate_diagnostics(self) -> dict[str, Any] | None:
        """JSON-safe report for the report-only selector-weight candidate."""

        if not self._strict_accessor_head_allowed("selector_weights_candidate"):
            return None

        weights, diagnostics = self._selector_weight_vector_from_columns(
            object_column="selector_weights_candidate",
            scalar_prefixes=("selector_weight_candidate_", "selector_weights_candidate_"),
        )
        diagnostics["has_usable_selector_weights_candidate"] = weights is not None
        return diagnostics

    # -- population family head (distribution phase) -------------------------
    def family_logits(self) -> np.ndarray | None:
        """Dense ``[n_features, 36]`` population-family logits, or ``None``."""
        if not self._strict_accessor_head_allowed("population_family"):
            return None
        if "population_family_logits" not in self._df.columns:
            return None
        mat = _stack_object_column(self._df["population_family_logits"], N_CANONICAL_FAMILIES)
        if mat is None:
            return None
        return self._dense_matrix(mat)

    # -- structural heads (T-DIAKRINO-IMP-05 trust probe) ---------------------------
    def population_dependency_predictions(self) -> np.ndarray | None:
        """Dense ``[n_features, 8]`` DIAKRINO dependency-head predictions, or ``None``.

        Channel layout mirrors ``experimental.population_ground_truth``:
        ``block_id_scaled``, ``block_loading_norm``, first two block-loading
        coordinates, max/mean absolute covariance-or-precision row summaries,
        scaled edge degree, and one canonicalized dependence scalar.
        """
        if not self._strict_accessor_head_allowed("population_dependency"):
            return None
        column = "population_dependency_predictions"
        if column not in self._df.columns:
            return None
        mat = _stack_object_column(self._df[column], POPULATION_DEPENDENCY_DIM)
        if mat is None:
            return None
        return self._dense_matrix(mat[:, :POPULATION_DEPENDENCY_DIM])

    def population_coeff_predictions(self) -> np.ndarray | None:
        """Dense ``[n_features, 4]`` DIAKRINO coefficient-head predictions, or ``None``.

        Channel layout mirrors ``experimental.population_ground_truth``:
        main/linear effect, interaction or higher-order accumulated weight,
        max polynomial order, and active-feature flag.
        """
        if not self._strict_accessor_head_allowed("population_coeff"):
            return None
        column = "population_coeff_predictions"
        if column not in self._df.columns:
            return None
        mat = _stack_object_column(self._df[column], POPULATION_COEFF_DIM)
        if mat is None:
            return None
        return self._dense_matrix(mat[:, :POPULATION_COEFF_DIM])

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.nanmax(logits, axis=-1, keepdims=True)
        e = np.exp(np.where(np.isfinite(z), z, -np.inf))
        s = np.sum(e, axis=-1, keepdims=True)
        return np.divide(e, s, out=np.zeros_like(e), where=s > 0)

    def family_entropy(self, *, normalized: bool = True) -> np.ndarray | None:
        """Per-feature softmax entropy of the family head (trust proxy).

        High entropy == the family head is undecided == low trust.  ``normalized``
        divides by ``ln(N_continuous)`` so the result is in ``[0, 1]``.  Returns
        ``None`` when the family head is absent.
        """
        fl = self.family_logits()
        if fl is None:
            return None
        cont = fl[:, :N_CONTINUOUS_FAMILIES]
        p = self._softmax(cont)
        ent = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=-1)
        if normalized:
            ent = ent / float(np.log(N_CONTINUOUS_FAMILIES))
        return ent

    def family_argmax_ids(self) -> np.ndarray | None:
        """Per-feature argmax family id over the FULL 36-way head (incl. discrete)."""
        fl = self.family_logits()
        if fl is None:
            return None
        safe = np.where(np.isfinite(fl), fl, -np.inf)
        return np.argmax(safe, axis=-1).astype(np.int64)

    def discrete_skip_mask(self) -> np.ndarray | None:
        """Boolean ``[n_features]``: True where the family argmax is a non-continuous
        id (>= 31, i.e. discrete or nuisance overflow) and the feature should bypass
        the continuous fit entirely (``skip_fit``)."""
        ids = self.family_argmax_ids()
        if ids is None:
            return None
        return ids >= N_CONTINUOUS_FAMILIES

    def family_topk_continuous(self, k: int = 4) -> list[list[int]] | None:
        """Per-feature top-``k`` CONTINUOUS family ids (0..30), discrete ids dropped.

        Used by the family-prescreen shortlist.  Always returns continuous-only ids so
        the downstream selector never receives an unmappable family.
        """
        fl = self.family_logits()
        if fl is None:
            return None
        cont = fl[:, :N_CONTINUOUS_FAMILIES]
        cont = np.where(np.isfinite(cont), cont, -np.inf)
        order = np.argsort(-cont, axis=-1)[:, : max(1, int(k))]
        return [[int(j) for j in row] for row in order]

    @staticmethod
    def family_id_to_name(family_id: int) -> str | None:
        """Map a family id to its canonical name, or ``None`` for out-of-range ids."""
        if 0 <= int(family_id) < N_CANONICAL_FAMILIES:
            return CANONICAL_MARGINAL_FAMILIES[int(family_id)]
        return None

    def family_signal_summary(self, *, top_k: int = 4) -> dict[str, Any] | None:
        """Dataset-level trusted family-head summary.

        Only the learned population-family head is read.  Discrete/nuisance ids
        (>=31) are counted before any id->continuous-family decode so downstream
        consumers never treat them as scipy families.
        """
        logits = self.family_logits()
        if logits is None:
            return None
        ids = self.family_argmax_ids()
        entropy = self.family_entropy(normalized=True)
        if ids is None or entropy is None:
            return None
        cont = np.asarray(logits[:, :N_CONTINUOUS_FAMILIES], dtype=np.float64)
        probs = self._softmax(cont)
        confidence = np.nanmax(probs, axis=1) if probs.size else np.asarray([], dtype=np.float64)
        ids_arr = np.asarray(ids, dtype=np.int64).ravel()
        continuous = (ids_arr >= 0) & (ids_arr < N_CONTINUOUS_FAMILIES)
        discrete = ids_arr >= N_CONTINUOUS_FAMILIES

        histogram: dict[str, int] = {}
        for family_id in ids_arr[continuous].tolist():
            name = self.family_id_to_name(int(family_id))
            if not name:
                continue
            histogram[str(name)] = int(histogram.get(str(name), 0) + 1)
        top = sorted(histogram.items(), key=lambda item: (-int(item[1]), str(item[0])))[: max(1, int(top_k))]

        ent = np.asarray(entropy, dtype=np.float64)
        conf = np.asarray(confidence, dtype=np.float64)
        finite_ent = ent[np.isfinite(ent)]
        finite_conf = conf[np.isfinite(conf)]
        n = int(ids_arr.size)
        return {
            "source": "diakrino_population_family_logits",
            "n_features": int(n),
            "continuous_feature_count": int(np.sum(continuous)),
            "discrete_skip_count": int(np.sum(discrete)),
            "discrete_skip_fraction": float(np.sum(discrete) / max(1, n)),
            "family_entropy_mean": float(np.mean(finite_ent)) if finite_ent.size else float("nan"),
            "family_entropy_median": float(np.median(finite_ent)) if finite_ent.size else float("nan"),
            "family_confidence_mean": float(np.mean(finite_conf)) if finite_conf.size else float("nan"),
            "family_confidence_median": float(np.median(finite_conf)) if finite_conf.size else float("nan"),
            "family_top1_histogram": {str(name): int(count) for name, count in sorted(histogram.items())},
            "family_top1_top": [{"family": str(name), "count": int(count)} for name, count in top],
            "top_k_requested": int(max(1, int(top_k))),
        }

    def selection_dispersion_summary(
        self,
        *,
        column: str = "feature_selection_logit",
        calibrate: str = "chunk_zscore",
    ) -> dict[str, Any] | None:
        """Cross-chunk dispersion descriptor from calibrated trusted selection logits."""
        column = str(column)
        if self._validated_identity is not None:
            trust_head = _strict_scalar_column_trust_head(column)
            if not trust_head or not self._strict_accessor_head_allowed(trust_head):
                return None
        if column not in self._df.columns:
            return None
        chunk_ids = self._chunk_ids()
        raw = np.asarray(self._df[column].to_numpy(), dtype=np.float64)
        scores = chunk_calibrate(raw, chunk_ids, str(calibrate or "chunk_zscore"))
        finite = np.isfinite(scores)
        if not np.any(finite):
            return None
        if chunk_ids is None:
            chunk_ids = np.zeros(scores.shape[0], dtype=np.int64)
        chunk_ids = np.asarray(chunk_ids).ravel()

        chunk_means: list[float] = []
        chunk_stds: list[float] = []
        chunk_q90: list[float] = []
        chunk_top_share: list[float] = []
        for chunk in np.unique(chunk_ids[finite]):
            mask = finite & (chunk_ids == chunk)
            if not np.any(mask):
                continue
            vals = np.asarray(scores[mask], dtype=np.float64)
            chunk_means.append(float(np.mean(vals)))
            chunk_stds.append(float(np.std(vals)))
            chunk_q90.append(float(np.quantile(vals, 0.90)))
            chunk_top_share.append(float(np.mean(vals >= float(np.quantile(scores[finite], 0.90)))))

        def _spread(values: list[float]) -> float:
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            return float(np.std(arr)) if arr.size else float("nan")

        return {
            "source": "diakrino_selection_logits",
            "score_column": str(column),
            "calibration": str(calibrate or "chunk_zscore"),
            "n_features": int(scores.shape[0]),
            "n_valid_scores": int(np.sum(finite)),
            "n_chunks": int(len(chunk_means)),
            "global_score_std": float(np.std(scores[finite])),
            "chunk_mean_std": _spread(chunk_means),
            "chunk_std_std": _spread(chunk_stds),
            "chunk_q90_std": _spread(chunk_q90),
            "chunk_top_decile_share_std": _spread(chunk_top_share),
        }

    def trusted_signal_summary(
        self,
        *,
        score_column: str = "feature_selection_logit",
        calibrate: str = "chunk_zscore",
    ) -> dict[str, Any]:
        """Combined report-only DIAKRINO signal summary for default-off consumers."""
        return {
            "family": self.family_signal_summary(),
            "selection_dispersion": self.selection_dispersion_summary(
                column=str(score_column),
                calibrate=str(calibrate or "chunk_zscore"),
            ),
        }

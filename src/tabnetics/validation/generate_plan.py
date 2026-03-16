#!/usr/bin/env python3
"""Generate a sharded validation plan (up to 20 pods) for ``tabnetics-validation-plan``, aligned with the promoted ``df_stage_position="after_fs"`` runtime and the stricter HuggingFace public-source mirror policy.

This produces:
- plan_<N>.json: structured job list
- shards_<N>.json: job assignment per shard
- WORK_SPLIT_<N>.md: human-readable mapping

The plan is intentionally derived from repo-local packaged sources so it stays in sync:
- FS method-set keys from `tabnetics.benchmarks.profiles`
- Deprecated validation exclusions from Progress.md (intersection with FS_METHOD_SETS)

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import numpy as np
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from tabnetics.core.paths import find_project_root, find_repo_root

CORE_PROJECT_ROOT = find_project_root(__file__)
REPO_ROOT = find_repo_root(__file__)
DEFAULT_MAX_PODS = 20
VALIDATION_SEEDS = [11, 23, 37, 42, 59, 67, 73, 89, 97]


@dataclass(frozen=True)
class Job:
    job_id: str
    kind: str
    params: Dict[str, Any]
    weight: float


@dataclass(frozen=True)
class BenchmarkProfile:
    profile_id: str
    fs_method_set: str
    extra_args: Tuple[str, ...] = tuple()
    weight_mult: float = 1.0
    notes: str = ""
    job_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TabArenaPlanProfile:
    profile_id: str
    profile: str
    dataset_sets: Tuple[str, ...] = tuple()
    datasets: Tuple[str, ...] = tuple()
    exclude_datasets: Tuple[str, ...] = tuple()
    seeds: Tuple[int, ...] = (42,)
    protocol: str = "openml_task"
    official_fold_limit: int = 0
    extra_args: Tuple[str, ...] = tuple()
    weight_mult: float = 1.0
    notes: str = ""
    quiet: bool = False
    skip_official_leaderboard: bool = True
    leaderboard_method_name: str = ""
    job_params: Dict[str, Any] = field(default_factory=dict)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_fs_method_sets() -> Dict[str, Tuple[str, ...]]:
    from tabnetics.benchmarks.profiles import FS_METHOD_SETS

    return dict(FS_METHOD_SETS)


def _load_benchmark_registry() -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    from tabnetics.datasets.benchmark_catalog import BENCHMARK_DATASETS, DATASET_SETS

    return dict(BENCHMARK_DATASETS), dict(DATASET_SETS)


def _load_tabarena_registry() -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    try:
        from experiments.benchmarking.tabarena_datasets import TABARENA_DATASETS, TABARENA_DATASET_SETS
    except ModuleNotFoundError:
        repo_root_txt = str(REPO_ROOT)
        if repo_root_txt not in sys.path:
            sys.path.insert(0, repo_root_txt)
        from experiments.benchmarking.tabarena_datasets import TABARENA_DATASETS, TABARENA_DATASET_SETS

    return dict(TABARENA_DATASETS), {str(k): list(v) for k, v in dict(TABARENA_DATASET_SETS).items()}


def _dataset_weight(spec: Any) -> float:
    """Rough runtime weight for a validation-catalog dataset."""
    tier = str(getattr(spec, "tier", "") or "").strip().lower()
    base = {
        "easy": 1.0,
        "medium": 1.3,
        "hard": 1.7,
        "very_hard": 2.2,
    }.get(tier, 1.2)
    pipeline = str(getattr(spec, "validation_pipeline", "") or "").strip().lower()
    if pipeline == "integrated":
        base += 0.3
    return float(base)


def _infer_dataset_class_count(dataset_id: str, benchmark_datasets: Mapping[str, Any]) -> Optional[int]:
    """Best-effort class-count lookup from benchmark metadata (follows base_dataset chain)."""
    seen: set[str] = set()
    current = str(dataset_id)
    while current and current not in seen:
        seen.add(current)
        spec = benchmark_datasets.get(current)
        if spec is None:
            return None
        params = getattr(spec, "params", {}) or {}
        if isinstance(params, dict):
            synth = params.get("synthetic_profile")
            if isinstance(synth, dict) and synth.get("n_classes") is not None:
                try:
                    return int(synth.get("n_classes"))
                except Exception:
                    pass
            if params.get("n_classes") is not None:
                try:
                    return int(params.get("n_classes"))
                except Exception:
                    pass
            base = params.get("base_dataset")
            if base:
                current = str(base)
                continue
        break
    return None


def _load_val4_runtime_hints(runs_csv: Path) -> Dict[str, Dict[str, float]]:
    """Load mean seed runtime (sec) by profile/dataset from Val-4 aggregate runs CSV."""
    if not runs_csv.exists():
        return {}

    buckets: Dict[Tuple[str, str], List[float]] = {}
    with runs_csv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("protocol", "")).strip() != "holdout":
                continue
            if str(row.get("config", "")).strip() != "baseline":
                continue
            job_id = str(row.get("job_id", "")).strip()
            parts = job_id.split("/")
            if len(parts) < 3:
                continue
            profile_id = str(parts[1])
            dataset_id = str(row.get("dataset_id", "")).strip()
            if not dataset_id:
                continue
            try:
                fs_t = float(row.get("fs_time_sec", "0") or 0.0)
                dist_t = float(row.get("dist_time_sec", "0") or 0.0)
                xfm_t = float(row.get("transform_time_sec", "0") or 0.0)
            except Exception:
                continue
            total_t = fs_t + dist_t + xfm_t
            key = (profile_id, dataset_id)
            buckets.setdefault(key, []).append(float(total_t))

    out: Dict[str, Dict[str, float]] = {}
    for (profile_id, dataset_id), vals in buckets.items():
        if not vals:
            continue
        out.setdefault(profile_id, {})[dataset_id] = float(sum(vals) / len(vals))
    return out


def _load_hf_manifest_metadata() -> Dict[str, Dict[str, Any]]:
    """Load optional dataset shape/class metadata from local HF bundle manifests."""
    candidates = [
        REPO_ROOT / "train_data" / "hf_expanded_bundle" / "manifest.json",
        REPO_ROOT / "train_data" / "hf_validation_bundle" / "manifest.json",
    ]
    out: Dict[str, Dict[str, Any]] = {}
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        datasets = payload.get("datasets") or {}
        if not isinstance(datasets, dict):
            continue
        for ds_id, raw in datasets.items():
            if not isinstance(raw, dict):
                continue
            rec = dict(out.get(str(ds_id)) or {})
            shape = raw.get("shape")
            if isinstance(shape, list) and len(shape) >= 2:
                try:
                    rec["n_samples"] = int(shape[0])
                    rec["n_features"] = int(shape[1])
                except Exception:
                    pass
            if raw.get("n_classes") is not None:
                try:
                    rec["n_classes"] = int(raw.get("n_classes"))
                except Exception:
                    pass
            if raw.get("tier"):
                rec["tier"] = str(raw.get("tier")).strip().lower()
            if raw.get("pipeline"):
                rec["pipeline"] = str(raw.get("pipeline")).strip().lower()
            out[str(ds_id)] = rec
    return out


def _estimate_val5_runtime_map(
    *,
    validation_ids: Sequence[str],
    benchmark_datasets: Mapping[str, Any],
    runtime_hints: Mapping[str, Mapping[str, float]],
    manifest_meta: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Estimate per-dataset runtime for Val-5 profiles.

    Val-5 profiles:
    - baseline: current production baseline
    - new_baseline_bcd: complete B+C+D stack (broad_val4 + oracle bundle + PLS-DA folding)
    """
    baseline_hist = dict(runtime_hints.get("baseline") or {})
    broad_hist = dict(runtime_hints.get("broad_oracle") or {})
    new_hist = dict(runtime_hints.get("new_methods") or {})
    manifest = dict(manifest_meta or {})

    def _median_or(vals: Sequence[float], default: float) -> float:
        clean = [float(v) for v in vals if float(v) > 0.0]
        if not clean:
            return float(default)
        clean = sorted(clean)
        return float(clean[len(clean) // 2])

    ratio_new_over_baseline: List[float] = []
    for ds_id, base_t in baseline_hist.items():
        new_t = new_hist.get(ds_id)
        if new_t is None:
            continue
        if base_t <= 1e-6:
            continue
        ratio_new_over_baseline.append(float(new_t / base_t))
    ratio_new = _median_or(ratio_new_over_baseline, 1.26)

    ratio_broad_over_baseline: List[float] = []
    ratio_new_over_broad: List[float] = []
    for ds_id, base_t in baseline_hist.items():
        broad_t = broad_hist.get(ds_id)
        if broad_t is not None and base_t > 1e-6:
            ratio_broad_over_baseline.append(float(broad_t / base_t))
    for ds_id, broad_t in broad_hist.items():
        new_t = new_hist.get(ds_id)
        if new_t is not None and broad_t > 1e-6:
            ratio_new_over_broad.append(float(new_t / broad_t))
    ratio_broad = _median_or(ratio_broad_over_baseline, 1.35)
    ratio_new_vs_broad = _median_or(ratio_new_over_broad, 1.05)

    # Build a baseline-proxy runtime table for any dataset observed in Val-4.
    baseline_proxy: Dict[str, float] = {}
    for ds_id in set(baseline_hist) | set(broad_hist) | set(new_hist):
        if ds_id in baseline_hist:
            baseline_proxy[ds_id] = float(max(1.0, baseline_hist[ds_id]))
        elif ds_id in broad_hist:
            baseline_proxy[ds_id] = float(max(1.0, broad_hist[ds_id] / max(1e-6, ratio_broad)))
        else:
            baseline_proxy[ds_id] = float(max(1.0, new_hist[ds_id] / max(1e-6, ratio_new)))

    # Tier/prefix priors from known datasets.
    tier_runtime_samples: Dict[str, List[float]] = {}
    tier_complexity_samples: Dict[str, List[float]] = {}
    prefix_runtime_samples: Dict[str, List[float]] = {}
    for ds_id, rt in baseline_proxy.items():
        spec = benchmark_datasets.get(ds_id)
        tier = str(getattr(spec, "tier", "") or "").strip().lower()
        if not tier and ds_id in manifest:
            tier = str((manifest.get(ds_id) or {}).get("tier") or "").strip().lower()
        if tier:
            tier_runtime_samples.setdefault(tier, []).append(float(rt))
        prefix = str(ds_id).split("_")[0]
        prefix_runtime_samples.setdefault(prefix, []).append(float(rt))
        meta = manifest.get(ds_id) or {}
        n = meta.get("n_samples")
        p = meta.get("n_features")
        if tier and isinstance(n, int) and isinstance(p, int) and n > 0 and p > 0:
            tier_complexity_samples.setdefault(tier, []).append(float(n) * float(p))

    tier_runtime_median = {
        tier: _median_or(vals, 260.0)
        for tier, vals in tier_runtime_samples.items()
        if vals
    }
    tier_complexity_median = {
        tier: _median_or(vals, 1.0e6)
        for tier, vals in tier_complexity_samples.items()
        if vals
    }
    prefix_runtime_median = {
        pref: _median_or(vals, 260.0)
        for pref, vals in prefix_runtime_samples.items()
        if vals
    }

    baseline_map: Dict[str, float] = {}
    bcd_map: Dict[str, float] = {}

    for ds_id in validation_ids:
        spec = benchmark_datasets.get(ds_id)
        fallback_seed_sec = float(_dataset_weight(spec) * 220.0)
        tier = str(getattr(spec, "tier", "") or "").strip().lower()
        prefix = str(ds_id).split("_")[0]
        meta = dict(manifest.get(str(ds_id)) or {})

        base_seed_sec = baseline_hist.get(ds_id)
        if base_seed_sec is None:
            # Fall back to other observed profiles if baseline for this dataset
            # is missing due partial Val-4 coverage.
            broad_seed = broad_hist.get(ds_id)
            if broad_seed is not None:
                base_seed_sec = float(max(1.0, broad_seed / max(1e-6, ratio_broad)))
        if base_seed_sec is None:
            new_seed = new_hist.get(ds_id)
            if new_seed is not None:
                base_seed_sec = float(max(1.0, new_seed / max(1e-6, ratio_new)))
        if base_seed_sec is None:
            # Shape-aware estimate for extended datasets not present in Val-4:
            # combine tier prior + prefix prior + complexity scaling from manifest shape.
            candidates: List[float] = [float(fallback_seed_sec)]
            tier_prior = tier_runtime_median.get(tier)
            if tier_prior is not None:
                candidates.append(float(tier_prior))
            pref_vals = prefix_runtime_samples.get(prefix) or []
            pref_prior = prefix_runtime_median.get(prefix)
            if pref_prior is not None and len(pref_vals) >= 2:
                candidates.append(float(pref_prior))

            n = meta.get("n_samples")
            p = meta.get("n_features")
            cls = meta.get("n_classes")
            if (
                tier_prior is not None
                and tier in tier_complexity_median
                and isinstance(n, int)
                and isinstance(p, int)
                and n > 0
                and p > 0
            ):
                complexity = float(n) * float(p)
                comp_ref = float(max(1.0, tier_complexity_median[tier]))
                comp_factor = float((complexity / comp_ref) ** 0.35)
                comp_factor = float(max(0.55, min(2.80, comp_factor)))
                cls_int = int(cls) if isinstance(cls, int) and cls > 0 else 2
                cls_factor = float((max(2, cls_int) / 2.0) ** 0.10)
                cls_factor = float(max(0.90, min(1.35, cls_factor)))
                shape_est = float(tier_prior * comp_factor * cls_factor)
                candidates.append(shape_est)
                if pref_prior is not None and len(pref_vals) >= 2:
                    candidates.append(float(0.60 * pref_prior + 0.40 * shape_est))

            base_seed_sec = _median_or(candidates, fallback_seed_sec)
        base_seed_sec = float(max(1.0, base_seed_sec))

        new_seed_sec = new_hist.get(ds_id)
        if new_seed_sec is None:
            if ds_id in broad_hist:
                new_seed_sec = float(max(1.0, broad_hist[ds_id] * ratio_new_vs_broad))
            else:
                new_seed_sec = float(max(1.0, base_seed_sec * ratio_new))

        # D overlay: PLS-DA folding only applies for >=5 classes.
        # Use a mild runtime factor for the active subset; otherwise keep unchanged.
        n_classes = None
        if isinstance(meta.get("n_classes"), int):
            n_classes = int(meta.get("n_classes"))
        if n_classes is None:
            n_classes = _infer_dataset_class_count(ds_id, benchmark_datasets)
        pls_factor = 1.08 if (n_classes is not None and n_classes >= 5) else 1.00
        bcd_seed_sec = float(max(1.0, new_seed_sec * pls_factor))

        baseline_map[str(ds_id)] = float(base_seed_sec)
        bcd_map[str(ds_id)] = float(bcd_seed_sec)

    return {"baseline": baseline_map, "new_baseline_bcd": bcd_map}


def _balanced_partition(items: Sequence[Tuple[str, float]], n_parts: int) -> List[List[str]]:
    """Greedy bin-pack items into n_parts lists by weight."""
    parts: List[List[str]] = [[] for _ in range(int(n_parts))]
    totals: List[float] = [0.0 for _ in range(int(n_parts))]
    weight_by_id: Dict[str, float] = {}
    for item_id, w in sorted(items, key=lambda kv: float(kv[1]), reverse=True):
        idx = int(min(range(len(totals)), key=lambda i: totals[i]))
        parts[idx].append(str(item_id))
        w_f = float(w)
        totals[idx] += w_f
        weight_by_id[str(item_id)] = w_f

    # Local search refinement: deterministic move/swap pass to reduce max/min spread.
    # This is small (<=67 datasets, <=12 parts in our usage), so a bounded O(N^2) search
    # per pass is fast and yields materially tighter shard balance.
    def _ratio(ts: Sequence[float]) -> float:
        min_t = float(min(ts)) if ts else 0.0
        max_t = float(max(ts)) if ts else 0.0
        return float(max_t / min_t) if min_t > 0 else float("inf")

    max_passes = 200
    for _ in range(max_passes):
        current = _ratio(totals)
        best_ratio = current
        best_op: Optional[Tuple[str, int, int, str, Optional[str], float, float]] = None

        # Candidate moves: move one item i -> j.
        for i in range(len(parts)):
            if not parts[i]:
                continue
            for item in parts[i]:
                wi = float(weight_by_id.get(item, 0.0))
                if wi <= 0.0:
                    continue
                for j in range(len(parts)):
                    if i == j:
                        continue
                    new_i = float(totals[i] - wi)
                    new_j = float(totals[j] + wi)
                    if new_i <= 0.0:
                        continue
                    trial = list(totals)
                    trial[i] = new_i
                    trial[j] = new_j
                    rr = _ratio(trial)
                    if rr + 1e-12 < best_ratio:
                        best_ratio = rr
                        best_op = ("move", i, j, item, None, new_i, new_j)

        # Candidate swaps: swap one item between i and j.
        for i in range(len(parts)):
            if not parts[i]:
                continue
            for j in range(i + 1, len(parts)):
                if not parts[j]:
                    continue
                for item_i in parts[i]:
                    wi = float(weight_by_id.get(item_i, 0.0))
                    if wi <= 0.0:
                        continue
                    for item_j in parts[j]:
                        wj = float(weight_by_id.get(item_j, 0.0))
                        if wj <= 0.0:
                            continue
                        new_i = float(totals[i] - wi + wj)
                        new_j = float(totals[j] - wj + wi)
                        if new_i <= 0.0 or new_j <= 0.0:
                            continue
                        trial = list(totals)
                        trial[i] = new_i
                        trial[j] = new_j
                        rr = _ratio(trial)
                        if rr + 1e-12 < best_ratio:
                            best_ratio = rr
                            best_op = ("swap", i, j, item_i, item_j, new_i, new_j)

        if best_op is None:
            break

        op, i, j, item_i, item_j, new_i, new_j = best_op
        if op == "move":
            parts[i].remove(item_i)
            parts[j].append(item_i)
            totals[i] = new_i
            totals[j] = new_j
            continue

        # swap
        assert item_j is not None
        idx_i = parts[i].index(item_i)
        idx_j = parts[j].index(item_j)
        parts[i][idx_i] = item_j
        parts[j][idx_j] = item_i
        totals[i] = new_i
        totals[j] = new_j
    return parts


def _load_deprecated_method_sets(fs_method_sets: Sequence[str]) -> List[str]:
    """Parse Progress.md for the 'Do Not Validate Again' line and return intersections."""
    progress_path = CORE_PROJECT_ROOT / "Progress.md"
    if not progress_path.exists():
        return []
    if not progress_path.exists():
        return []

    text = progress_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Do not include in future validation runs:(.*)", text)
    if not m:
        return []

    tokens = [t.strip() for t in re.findall(r"`([^`]+)`", m.group(1))]
    fs_set = set(fs_method_sets)
    # Ignore doc references etc (keep only known method-set keys).
    out = [t for t in tokens if t in fs_set]
    return sorted(set(out))


def _estimate_benchmark_weight(method_set: str) -> float:
    base = 1000.0
    if method_set.startswith("a") and method_set[1:].isdigit() is False:
        # "a22_nsc" etc are still full sweeps; keep default weight.
        pass
    if method_set.startswith("a") and any(x in method_set for x in ("nsc", "hsic", "pareto")):
        base -= 250.0  # standalone single-method portfolios tend to be faster

    bumps = [
        ("copula", 450.0),
        ("hsic", 450.0),
        ("iterative_pruning", 350.0),
        ("decorrelated", 250.0),
        ("tigress", 200.0),
        ("ipss", 200.0),
        ("cluster", 150.0),
        ("sparse_multinomial", 150.0),
    ]
    for key, bump in bumps:
        if key in method_set:
            base += bump
    return float(max(200.0, base))


def _balanced_shard_assign(jobs: Sequence[Job], num_shards: int) -> Dict[int, List[str]]:
    """Greedy bin-pack by weight (good enough for 20 shards)."""
    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for job in sorted(jobs, key=lambda j: float(j.weight), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        shards[target].append(job.job_id)
        totals[target] += float(job.weight)
    return shards


def _balanced_shard_assign_validation5_pairs(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-5 by paired dataset partitions (baseline + new_baseline_bcd)."""
    pair_re = re.compile(r"^val5/(baseline|new_baseline_bcd)/(ds\d+)$")
    paired: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = pair_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        paired.setdefault(part_id, {})[profile_id] = job

    pair_items: List[Tuple[float, Job, Job]] = []
    for part_id, bundle in paired.items():
        baseline_job = bundle.get("baseline")
        new_job = bundle.get("new_baseline_bcd")
        if baseline_job is None or new_job is None:
            # Fallback safety: keep any partial profile entries in the unpaired pool.
            for partial in bundle.values():
                unpaired.append(partial)
            continue
        pair_weight = float(baseline_job.weight) + float(new_job.weight)
        pair_items.append((pair_weight, new_job, baseline_job))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for pair_weight, new_job, baseline_job in sorted(pair_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        # Keep profile order consistent in shard docs and run logs.
        shards[target].append(new_job.job_id)
        shards[target].append(baseline_job.job_id)
        totals[target] += float(pair_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


def _balanced_shard_assign_validation6_pairs(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-6 by paired dataset partitions (baseline + full)."""
    pair_re = re.compile(r"^val6/(baseline|full)/(ds\d+)$")
    paired: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = pair_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        paired.setdefault(part_id, {})[profile_id] = job

    pair_items: List[Tuple[float, Job, Job]] = []
    for part_id, bundle in paired.items():
        baseline_job = bundle.get("baseline")
        full_job = bundle.get("full")
        if baseline_job is None or full_job is None:
            for partial in bundle.values():
                unpaired.append(partial)
            continue
        pair_weight = float(baseline_job.weight) + float(full_job.weight)
        pair_items.append((pair_weight, full_job, baseline_job))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for pair_weight, full_job, baseline_job in sorted(pair_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        shards[target].append(full_job.job_id)
        shards[target].append(baseline_job.job_id)
        totals[target] += float(pair_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


def _shard_weight_totals(shards: Dict[int, List[str]], jobs_by_id: Dict[str, Job]) -> Dict[int, float]:
    totals: Dict[int, float] = {}
    for shard_id, job_ids in shards.items():
        totals[int(shard_id)] = float(
            sum(float(jobs_by_id[jid].weight) for jid in job_ids if jid in jobs_by_id)
        )
    return totals


def _job(
    job_id: str,
    kind: str,
    *,
    weight: float,
    **params: Any,
) -> Job:
    return Job(job_id=job_id, kind=kind, params=dict(params), weight=float(weight))


@lru_cache(maxsize=1)
def _load_implicit_true_negation_rules() -> Dict[str, Dict[str, Tuple[str, ...]]]:
    """Discover argparse toggles that default to True and require explicit negation."""
    from tabnetics.benchmarks.runner import build_arg_parser

    parser = build_arg_parser()
    positive_by_dest: Dict[str, List[str]] = {}
    negative_by_dest: Dict[str, List[str]] = {}

    for action in getattr(parser, "_actions", []):
        option_strings = [
            str(flag)
            for flag in getattr(action, "option_strings", ())
            if str(flag).startswith("--")
        ]
        if not option_strings:
            continue

        dest = str(getattr(action, "dest", "") or "")
        if not dest:
            continue

        action_name = type(action).__name__
        default = getattr(action, "default", None)
        if action_name == "_StoreTrueAction" and default is True:
            positive_by_dest.setdefault(dest, []).extend(option_strings)
        elif action_name == "_StoreFalseAction":
            negative_by_dest.setdefault(dest, []).extend(option_strings)

    rules: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    for dest, pos_flags in positive_by_dest.items():
        neg_flags = tuple(sorted({str(f) for f in negative_by_dest.get(dest, []) if str(f).startswith("--no-")}))
        if not neg_flags:
            continue
        rules[str(dest)] = {
            "positive_flags": tuple(sorted({str(f) for f in pos_flags})),
            "negation_flags": neg_flags,
        }
    return rules


def _assert_no_implicit_true_omissions(
    *,
    profiles: Sequence[BenchmarkProfile],
    required_negations_by_profile: Optional[Mapping[str, Sequence[str]]] = None,
    required_value_overrides_by_profile: Optional[Mapping[str, Sequence[Tuple[str, str]]]] = None,
    context: str,
) -> None:
    """Hard-fail when an ablation relies on omitting flags that default to True."""
    required = {
        str(k): tuple(str(v) for v in vals)
        for k, vals in dict(required_negations_by_profile or {}).items()
    }
    required_values = {
        str(k): tuple((str(flag), str(val)) for flag, val in vals)
        for k, vals in dict(required_value_overrides_by_profile or {}).items()
    }
    implicit_rules = _load_implicit_true_negation_rules()
    errors: List[str] = []

    for prof in profiles:
        pid = str(prof.profile_id)
        token_list = [str(tok) for tok in tuple(prof.extra_args)]
        token_set = set(token_list)

        for req_flag in required.get(pid, tuple()):
            if req_flag not in token_set:
                errors.append(
                    f"{pid}: missing required negation flag {req_flag!r} "
                    f"(implicit-True omission risk)."
                )

        for req_flag, req_value in required_values.get(pid, tuple()):
            idxs = [i for i, tok in enumerate(token_list) if tok == req_flag]
            if not idxs:
                errors.append(
                    f"{pid}: missing required value-override flag {req_flag!r}={req_value!r}."
                )
                continue
            last_idx = int(idxs[-1])
            if (last_idx + 1) >= len(token_list) or str(token_list[last_idx + 1]).startswith("--"):
                errors.append(
                    f"{pid}: flag {req_flag!r} is present but missing an explicit value "
                    f"(expected {req_value!r})."
                )
                continue
            observed = str(token_list[last_idx + 1])
            if observed != req_value:
                errors.append(
                    f"{pid}: last {req_flag!r} value is {observed!r}, expected {req_value!r}."
                )

        if "no_bh" in pid and "--no-prefilter-bh-ttest" not in token_set:
            errors.append(f"{pid}: profile id implies BH ablation but --no-prefilter-bh-ttest is missing.")
        if "no_varfloor" in pid and "--no-prefilter-variance-floor" not in token_set:
            errors.append(
                f"{pid}: profile id implies variance-floor ablation but --no-prefilter-variance-floor is missing."
            )
        if "no_extreme_multiclass" in pid and "--no-regime-gating-extreme-multiclass" not in token_set:
            errors.append(
                f"{pid}: profile id implies extreme-multiclass ablation but "
                "--no-regime-gating-extreme-multiclass is missing."
            )
        if "gates_only" in pid:
            if "--no-prefilter-bh-ttest" not in token_set:
                errors.append(
                    f"{pid}: gates-only profile must explicitly disable BH via --no-prefilter-bh-ttest."
                )
            if "--no-prefilter-variance-floor" not in token_set:
                errors.append(
                    f"{pid}: gates-only profile must explicitly disable variance floor via "
                    "--no-prefilter-variance-floor."
                )

        for dest, rule in implicit_rules.items():
            pos_flags = set(rule.get("positive_flags", tuple()))
            neg_flags = set(rule.get("negation_flags", tuple()))
            has_pos = bool(pos_flags & token_set)
            has_neg = bool(neg_flags & token_set)
            if has_pos and has_neg:
                last_pos = max((i for i, tok in enumerate(token_list) if tok in pos_flags), default=-1)
                last_neg = max((i for i, tok in enumerate(token_list) if tok in neg_flags), default=-1)
                if last_pos > last_neg:
                    errors.append(
                        f"{pid}: negation for implicit-True toggle {dest!r} is overridden by a later positive flag "
                        f"(pos_index={last_pos}, neg_index={last_neg})."
                    )

    if errors:
        raise RuntimeError(
            f"Implicit-True ablation omission checks failed in {context}:\n- " + "\n- ".join(errors)
        )


def build_jobs_legacy(*, include_feature_smoke: bool = True, include_validation_suite: bool = True) -> List[Job]:
    fs_method_sets = _load_fs_method_sets()
    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))

    # Known caveat: this preset is explicitly binary-only and will fail on many catalog datasets.
    excluded = {"strict_plus_mrmr_auc_joint_l1"}

    baseline_method_sets = [
        k
        for k in fs_method_sets.keys()
        if k not in deprecated and k not in excluded
    ]

    jobs: List[Job] = []

    # ------------------------- Full method-set sweeps -------------------------
    for method_set in baseline_method_sets:
        jobs.append(
            _job(
                f"bench_full/{method_set}",
                "run_df_fs_sota_benchmark",
                weight=_estimate_benchmark_weight(method_set),
                dataset_sets=["validation_all"],
                seeds=[11, 23, 37],
                fs_method_set=method_set,
                ablation_profile="none",
                allow_synthetic_fallback=False,
                dataset_integrity_policy="skip",
                quiet_worker_logs=True,
                progress_heartbeat_sec=30,
                progress_watchdog_sec=0,
                task_timeout_sec=0,
                fs_method_timeout_sec=0,
            )
        )

    # ------------------------- Feature smoke configs --------------------------
    if include_feature_smoke:
        # Keep these small: exercise opt-in code paths on representative subsets.
        # All of these flags exist today in `run_df_fs_sota_benchmark.py`.
        smoke_common = dict(
            seeds=[11],
            ablation_profile="none",
            allow_synthetic_fallback=False,
            dataset_integrity_policy="skip",
            quiet_worker_logs=True,
            progress_heartbeat_sec=30,
            progress_watchdog_sec=0,
            task_timeout_sec=0,
            fs_method_timeout_sec=0,
        )

        def add_smoke(job_suffix: str, *, dataset_sets: Sequence[str], fs_method_set: str, extra_args: Sequence[str], weight: float = 120.0) -> None:
            jobs.append(
                _job(
                    f"bench_smoke/{job_suffix}",
                    "run_df_fs_sota_benchmark",
                    weight=weight,
                    dataset_sets=list(dataset_sets),
                    fs_method_set=fs_method_set,
                    extra_args=list(extra_args),
                    **smoke_common,
                )
            )

        add_smoke(
            "a23_pls_da_folding",
            dataset_sets=["smoke"],
            fs_method_set="mnpo_joint_multiclass_extended",
            extra_args=["--folding-method", "pls_da"],
        )
        add_smoke(
            "a21_face_domain_projection",
            dataset_sets=["quick_non_genomics_hdlss"],
            fs_method_set="mnpo_joint_multiclass_extended",
            extra_args=["--enable-face-domain-projection"],
        )

        # A20-R1 screening modes (sparse multinomial).
        for mode in ("prefilter_aggressive", "prefilter_balanced", "prefilter_conservative"):
            add_smoke(
                f"a20r1_sparse_screening_{mode}",
                dataset_sets=["smoke"],
                fs_method_set="mnpo_sparse_multinomial_extended",
                extra_args=["--fs-sparse-multinomial-screening-mode", mode],
            )

        # A26 per-class quota overlay.
        add_smoke(
            "a26_nsc_per_class_quota",
            dataset_sets=["smoke"],
            fs_method_set="mnpo_nsc_extended",
            extra_args=["--enable-fs-per-class-quota"],
        )
        add_smoke(
            "a26_class_pareto_per_class_quota",
            dataset_sets=["smoke"],
            fs_method_set="mnpo_class_pareto_extended",
            extra_args=["--enable-fs-per-class-quota"],
        )

        # A27 NSC threshold variants.
        for mode in ("hard", "order", "auto"):
            add_smoke(
                f"a27_nsc_thresholding_{mode}",
                dataset_sets=["smoke"],
                fs_method_set="mnpo_nsc_threshold_variants_extended",
                extra_args=["--fs-nsc-thresholding-mode", mode],
            )

        # CDF block-gating CV (integrated scenarios).
        add_smoke(
            "cdf_block_gating_cv",
            dataset_sets=["quick_integrated"],
            fs_method_set="mnpo_joint_multiclass_extended",
            extra_args=["--enable-cdf-block-gating-cv"],
            weight=220.0,
        )

        # Copula truncation (exercise truncation plumbing).
        add_smoke(
            "copula_truncation_level_10",
            dataset_sets=["smoke"],
            fs_method_set="mnpo_copula_extended",
            extra_args=["--fs-copula-truncation-level", "10"],
            weight=220.0,
        )

        # IPSS EATS calibration path.
        add_smoke(
            "ipss_eats_threshold_gb",
            dataset_sets=["smoke"],
            fs_method_set="mnpo_ipss_extended",
            extra_args=[
                "--enable-fs-ipss-eats-threshold",
                "--fs-ipss-importance-model",
                "gradient_boosting",
            ],
            weight=220.0,
        )

        # Stability loss-guided validation.
        add_smoke(
            "stability_loss_guided_validation",
            dataset_sets=["smoke"],
            fs_method_set="mnpo_extended",
            extra_args=["--enable-fs-stability-loss-guided-validation"],
            weight=200.0,
        )

        # SOTA-matched classifier mapping (runs extra config per dataset; keep it tiny).
        add_smoke(
            "sota_matched_classifiers",
            dataset_sets=["quick_easy_guard"],
            fs_method_set="strict_plus_mrmr",
            extra_args=["--use-sota-matched-classifiers"],
            weight=240.0,
        )

    # ---------------- Optional heavier / GPU-backed model candidates ----------
    #
    # These are intentionally not part of the default plan because they add
    # runtime/cost and require optional deps (tabpfn/xgboost). When run on GPU
    # pods, TabPFN will typically use CUDA automatically (device=auto).
    if bool(os.environ.get("PODVAL_INCLUDE_GPU_MODEL_SMOKE", "").strip()):
        smoke_common = dict(
            seeds=[11],
            ablation_profile="none",
            allow_synthetic_fallback=False,
            dataset_integrity_policy="skip",
            quiet_worker_logs=True,
            progress_heartbeat_sec=30,
            progress_watchdog_sec=0,
            task_timeout_sec=0,
            fs_method_timeout_sec=0,
        )

        jobs.append(
            _job(
                "bench_smoke/tabpfn_smoke",
                "run_df_fs_sota_benchmark",
                weight=420.0,
                dataset_sets=["smoke"],
                fs_method_set="mnpo_joint_multiclass_extended",
                extra_args=["--include-tabpfn-model"],
                **smoke_common,
            )
        )
        jobs.append(
            _job(
                "bench_smoke/xgb_quick_artificial_hdlss",
                "run_df_fs_sota_benchmark",
                weight=240.0,
                dataset_sets=["quick_artificial_hdlss"],
                fs_method_set="mnpo_joint_multiclass_extended",
                extra_args=["--include-xgb-model"],
                **smoke_common,
            )
        )

    # -------------------- Validation-suite DF coverage ------------------------
    if include_validation_suite:
        jobs.append(
            _job(
                "validation_suite/df_all_baseline",
                "validation_suite",
                weight=650.0,
                pipelines=["df"],
                dataset_sets=["df_all"],
                seeds=[11, 23, 37],
                ablation_profile="none",
                allow_synthetic_fallback=False,
                dataset_integrity_policy="skip",
            )
        )
        # Small component-toggle smoke (one dataset per pipeline).
        for pipeline in ("fs", "df", "integrated"):
            jobs.append(
                _job(
                    f"validation_suite/{pipeline}_smoke_single_toggle",
                    "validation_suite",
                    weight=180.0,
                    pipelines=[pipeline],
                    dataset_sets=["smoke"],
                    seeds=[11],
                    ablation_profile="single_toggle",
                    allow_synthetic_fallback=False,
                    dataset_integrity_policy="skip",
                )
            )

    return jobs


def build_jobs_validation1(*, dataset_shards: int = 8) -> List[Job]:
    """Validation-1: full-catalog 5-seed runs for baseline + single-feature toggles.

    This plan is designed to support an empirical feature-by-feature review against
    the production baseline and to update CurrentStatistics.md after aggregation.
    """
    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, dataset_sets = _load_benchmark_registry()

    validation_ids = list(dataset_sets.get("validation_all") or [])
    if not validation_ids:
        raise RuntimeError("No validation_all datasets found in benchmark registry.")

    baseline_method_set = "mnpo_joint_multiclass_extended"
    if baseline_method_set not in fs_method_sets:
        raise RuntimeError(
            f"Missing baseline method set {baseline_method_set!r} in FS_METHOD_SETS."
        )

    # Greedy weighted partitioning of datasets into dataset_shards parts.
    ds_items: List[Tuple[str, float]] = []
    ds_weight_by_id: Dict[str, float] = {}
    for ds_id in validation_ids:
        spec = benchmark_datasets.get(ds_id)
        w = _dataset_weight(spec) if spec is not None else 1.2
        ds_items.append((ds_id, w))
        ds_weight_by_id[str(ds_id)] = float(w)
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_job_params = dict(
        dataset_sets=[],
        seeds=list(VALIDATION_SEEDS),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        task_timeout_sec=0,
        fs_method_timeout_sec=0,
    )
    common_extra_args = ("--emit-summary", "--compute-budget", "standard")

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            profile_id="baseline",
            fs_method_set=baseline_method_set,
            extra_args=tuple(),
            weight_mult=1.0,
            notes="Production baseline (CurrentStatistics canonical).",
        ),
        # --- DF feature toggles ---
        BenchmarkProfile(
            profile_id="df_fastpath_off",
            fs_method_set=baseline_method_set,
            extra_args=("--disable-df-fastpath",),
            weight_mult=1.25,
            notes="A6 ablation: DF fast-path disabled.",
        ),
        BenchmarkProfile(
            profile_id="df_mnpo_oracle_fastpath_off",
            fs_method_set=baseline_method_set,
            extra_args=("--disable-df-fastpath", "--dist-criterion", "mnpo_oracle"),
            weight_mult=1.55,
            notes="T-008: dist_criterion=mnpo_oracle with DF fast-path disabled.",
        ),
        # --- FS feature toggles ---
        BenchmarkProfile(
            profile_id="fs_eval_proxy",
            fs_method_set=baseline_method_set,
            extra_args=(
                "--eval-models-enabled",
                "--eval-models",
                "lr_l2,linear_svc,rf_small",
                "--eval-aggregate",
                "mean",
            ),
            weight_mult=2.75,
            notes="T-001: multi-classifier evaluation proxy (fixed-weight aggregation).",
        ),
        BenchmarkProfile(
            profile_id="fs_eval_proxy_multi_model_oracles",
            fs_method_set=baseline_method_set,
            extra_args=(
                "--eval-models-enabled",
                "--eval-models",
                "lr_l2,linear_svc,rf_small",
                "--eval-aggregate",
                "mean",
                "--mnpo-performance-oracle-mode",
                "multi_model_oracles",
            ),
            weight_mult=2.85,
            notes="T-002: per-model performance oracles (requires eval proxy).",
        ),
        BenchmarkProfile(
            profile_id="fs_screening_stir",
            fs_method_set=baseline_method_set,
            extra_args=("--screening-enabled", "--screening-method", "stir"),
            weight_mult=1.60,
            notes="T-004: Tier-2 interaction-aware screening (STIR).",
        ),
        BenchmarkProfile(
            profile_id="fs_oracle_tail_risk",
            fs_method_set=baseline_method_set,
            extra_args=("--fs-use-tail-risk-oracle", "--fs-tail-risk-alpha", "0.33"),
            weight_mult=1.15,
            notes="MNPO oracle extension: tail-risk (CVaR) oracle (FS).",
        ),
        BenchmarkProfile(
            profile_id="fs_oracle_regret",
            fs_method_set=baseline_method_set,
            extra_args=("--fs-use-regret-oracle",),
            weight_mult=1.15,
            notes="MNPO oracle extension: fold-regret oracle (FS).",
        ),
        BenchmarkProfile(
            profile_id="fs_oracle_qre_smoothing",
            fs_method_set=baseline_method_set,
            extra_args=("--fs-use-qre-smoothing", "--fs-qre-temperature-gamma", "1.0"),
            weight_mult=1.15,
            notes="MNPO oracle extension: QRE smoothing for scalar-oracle preferences (FS).",
        ),
        BenchmarkProfile(
            profile_id="fs_oracle_redundancy_penalty",
            fs_method_set=baseline_method_set,
            extra_args=("--fs-use-oracle-redundancy-penalty",),
            weight_mult=1.15,
            notes="MNPO oracle extension: oracle-redundancy penalty (FS).",
        ),
        BenchmarkProfile(
            profile_id="prefilter_mi80_f20",
            fs_method_set=baseline_method_set,
            extra_args=("--prefilter-mi-weight", "0.8", "--prefilter-f-weight", "0.2"),
            weight_mult=1.05,
            notes="T-003 sensitivity: MI-heavy prefilter blend.",
        ),
        BenchmarkProfile(
            profile_id="prefilter_mi20_f80",
            fs_method_set=baseline_method_set,
            extra_args=("--prefilter-mi-weight", "0.2", "--prefilter-f-weight", "0.8"),
            weight_mult=1.05,
            notes="T-003 sensitivity: F-test-heavy prefilter blend.",
        ),
        # --- Deferred revalidation sweep (top candidates vs baseline) ---
        BenchmarkProfile(
            profile_id="candidate_a23_pls_da",
            fs_method_set=baseline_method_set,
            extra_args=("--folding-method", "pls_da"),
            weight_mult=1.35,
            notes="Revalidation sweep: A23 PLS-DA folding config (baseline method-set + folding).",
        ),
        BenchmarkProfile(
            profile_id="candidate_mnpo_dove_extended",
            fs_method_set="mnpo_dove_extended",
            extra_args=tuple(),
            weight_mult=1.05,
            notes="Revalidation sweep: method-set mnpo_dove_extended.",
        ),
        BenchmarkProfile(
            profile_id="candidate_a26_class_pareto_quota",
            fs_method_set="mnpo_class_pareto_extended",
            extra_args=("--enable-fs-per-class-quota",),
            weight_mult=1.10,
            notes="Revalidation sweep: class-Pareto with per-class quota overlay.",
        ),
        BenchmarkProfile(
            profile_id="candidate_a27_nsc_threshold_auto",
            fs_method_set="mnpo_nsc_threshold_variants_extended",
            extra_args=("--fs-nsc-thresholding-mode", "auto"),
            weight_mult=1.05,
            notes="Revalidation sweep: NSC thresholding auto (threshold-variants method-set).",
        ),
    ]

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError("Invalid validation1 profile(s):\n- " + "\n- ".join(invalid))

    jobs: List[Job] = []
    base_unit = 200.0
    for prof in profiles:
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_weight = sum(float(ds_weight_by_id.get(ds_id, 1.2)) for ds_id in ds_list)
            jobs.append(
                _job(
                    f"val1/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=float(base_unit * part_weight * float(prof.weight_mult)),
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    **common_job_params,
                )
            )

    # DF-only catalog coverage for DF MNPO criterion parity checking.
    jobs.append(
        _job(
            "val1/validation_suite/df_all_simple",
            "validation_suite",
            weight=450.0,
            pipelines=["df"],
            dataset_sets=["df_all"],
            seeds=list(VALIDATION_SEEDS),
            ablation_profile="none",
            allow_synthetic_fallback=False,
            dataset_integrity_policy="skip",
            df_criterion="simple",
        )
    )
    jobs.append(
        _job(
            "val1/validation_suite/df_all_mnpo_oracle",
            "validation_suite",
            weight=520.0,
            pipelines=["df"],
            dataset_sets=["df_all"],
            seeds=list(VALIDATION_SEEDS),
            ablation_profile="none",
            allow_synthetic_fallback=False,
            dataset_integrity_policy="skip",
            df_criterion="mnpo_oracle",
        )
    )

    return jobs


def build_jobs_validation4(*, dataset_shards: int = 6) -> List[Job]:
    """Validation-4: 3-profile decomposition + PLS-DA guardrail on 6 pods.

    Profile design (each shard runs all profiles for its dataset partition):

      A  baseline        mnpo_class_pareto_extended (7 methods)  — Val-3 regression anchor
      B  broad_oracle    mnpo_broad_stable (14 methods) + targeted oracle improvements
                         (paradigm prior, QRE smoothing, oracle redundancy penalty,
                          mRMR MI redundancy, prefilter union mi_ftest_blend+rf_importance,
                          e-value screening)
      C  new_methods     mnpo_broad_val4 (16 methods = stable + cmim + fcbf) + B's oracle
                         settings + multi-model oracles + copula derand K=5
      D  pls_da_guard    mnpo_class_pareto_extended + PLS-DA folding, 3 seeds only (advisory)

    Effect decomposition:
      B vs A  — method expansion (7→14) + oracle tuning (baseline comparison)
      C vs B  — CMIM/FCBF + multi-model oracles + derand K=5 incremental
      C vs A  — total Phase-2/2.5 delta (SOTA comparison anchor)
      D vs A  — PLS-DA guardrail regression (T-R-121)

    Deliberate exclusions (with rationale):
      --enable-fs-runtime-racing    : systematic confounder; cuts methods by wall time →
                                      different method subsets complete per pod
      --enable-fs-rashomon          : untested at scale on broad stack; memory unknown
      --enable-fs-wrapper-refine    : same; defer to post-Val-4
      --fs-copula-derandomize-runs 20: 20× knockoff cost without isolated baseline;
                                       K=5 budgets 5× cost with bounded risk
      --prefilter-strategies relieff_scores: triples prefilter cost; start with 2-strategy union
    """
    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, dataset_sets = _load_benchmark_registry()

    validation_ids = list(dataset_sets.get("validation_all") or [])
    if not validation_ids:
        raise RuntimeError("No validation_all datasets found in benchmark registry.")

    for required in ("mnpo_class_pareto_extended", "mnpo_broad_stable", "mnpo_broad_val4"):
        if required not in fs_method_sets:
            raise RuntimeError(f"Missing required method set {required!r} in FS_METHOD_SETS.")

    # Tier-aware weighted partition of datasets into shards.
    ds_items: List[Tuple[str, float]] = []
    ds_weight_by_id: Dict[str, float] = {}
    for ds_id in validation_ids:
        spec = benchmark_datasets.get(ds_id)
        w = _dataset_weight(spec) if spec is not None else 1.2
        ds_items.append((ds_id, w))
        ds_weight_by_id[str(ds_id)] = float(w)
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_job_params = dict(
        seeds=list(VALIDATION_SEEDS),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        task_timeout_sec=0,
        fs_method_timeout_sec=0,
    )
    common_extra_args: Tuple[str, ...] = ("--emit-summary", "--compute-budget", "standard")

    # Oracle flags shared by profiles B and C (targeted subset; no confounders).
    _oracle_flags: Tuple[str, ...] = (
        "--enable-fs-mnpo-paradigm-aware-prior",
        "--fs-use-qre-smoothing",
        "--fs-use-oracle-redundancy-penalty",
        "--fs-compute-tremble-sensitivity",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance",
        "--screening-enabled",
        "--screening-method", "evalue",
    )

    profiles: List[BenchmarkProfile] = [
        # ── Profile A: production baseline ─────────────────────────────────────
        BenchmarkProfile(
            profile_id="baseline",
            fs_method_set="mnpo_class_pareto_extended",
            extra_args=tuple(),
            weight_mult=1.0,
            notes="Val-4 baseline: production defaults (7 methods, no new toggles). "
                  "Paired comparison anchor vs Val-3.",
        ),
        # ── Profile B: method expansion + oracle tuning (no new algo methods) ──
        BenchmarkProfile(
            profile_id="broad_oracle",
            fs_method_set="mnpo_broad_stable",
            extra_args=_oracle_flags,
            # 14 methods ≈1.6× cost; prefilter union +10%; e-value screening +5%;
            # oracle flags negligible → net ≈1.8× baseline.
            weight_mult=1.80,
            notes="Val-4 candidate B: mnpo_broad_stable (14 methods) + oracle tuning "
                  "(paradigm prior, QRE, redundancy penalty, mRMR MI, prefilter union, "
                  "e-value screening). Excludes new algo methods. Tests method expansion "
                  "and oracle tuning in isolation.",
        ),
        # ── Profile C: new algo methods + multi-model oracle + derand knockoffs ─
        BenchmarkProfile(
            profile_id="new_methods",
            fs_method_set="mnpo_broad_val4",
            extra_args=(
                *_oracle_flags,
                # Multi-classifier oracle (T-R-132 validated wiring).
                "--eval-models-enabled",
                "--mnpo-performance-oracle-mode", "multi_model_oracles",
                "--eval-models", "lr_l2,linear_svc,rf_small",
                "--eval-aggregate", "mean",
                # Derandomized knockoffs K=5 (cost-bounded; not K=20).
                "--fs-copula-derandomize-runs", "5",
            ),
            # 16 methods ≈1.7× base; multi-model oracles ≈+40%; derand K=5 ≈+10%;
            # oracle flags as B → net ≈2.5× baseline.
            weight_mult=2.50,
            notes="Val-4 candidate C: mnpo_broad_val4 (16 methods = stable + cmim + fcbf) "
                  "+ B oracle settings + multi-model oracles + derand K=5. "
                  "Tests CMIM/FCBF and multi-model oracle incremental contribution over B.",
        ),
        # ── Profile D: PLS-DA guardrail (advisory; T-R-121) ─────────────────────
        BenchmarkProfile(
            profile_id="pls_da_guard",
            fs_method_set="mnpo_class_pareto_extended",
            extra_args=("--folding-method", "pls_da"),
            weight_mult=0.85,  # 3/5 seeds × 1.35 PLS-DA overhead ≈ 0.81, round up
            notes="Val-4 advisory D: PLS-DA guardrail (T-R-121). "
                  "3 seeds only (11 23 37); uses folding_pls_min_classes=5 default. "
                  "Not a promotion candidate — regression check only.",
        ),
    ]

    # advisory_profile_ids use only 3 seeds (cost containment)
    advisory_profiles = {"pls_da_guard"}
    pls_seeds = [11, 23, 37]

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError("Invalid validation4 profile(s):\n- " + "\n- ".join(invalid))

    jobs: List[Job] = []
    base_unit = 200.0

    for prof in profiles:
        seeds = pls_seeds if prof.profile_id in advisory_profiles else list(VALIDATION_SEEDS)
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_weight = sum(float(ds_weight_by_id.get(ds_id, 1.2)) for ds_id in ds_list)
            jobs.append(
                _job(
                    f"val4/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=float(base_unit * part_weight * float(prof.weight_mult)),
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    seeds=seeds,
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    ablation_profile="none",
                    allow_synthetic_fallback=False,
                    dataset_integrity_policy="skip",
                    quiet_worker_logs=bool(common_job_params["quiet_worker_logs"]),
                    progress_heartbeat_sec=int(common_job_params["progress_heartbeat_sec"]),
                    progress_watchdog_sec=int(common_job_params["progress_watchdog_sec"]),
                    task_timeout_sec=int(common_job_params["task_timeout_sec"]),
                    fs_method_timeout_sec=int(common_job_params["fs_method_timeout_sec"]),
                )
            )

    return jobs


def build_jobs_validation5(*, dataset_shards: int = 6, val4_root: Optional[Path] = None) -> List[Job]:
    """Validation-5: baseline anchor vs complete B+C+D candidate on 6 pods.

    Profiles:
      A  baseline          mnpo_class_pareto_extended (production anchor, 5 seeds)
      E  new_baseline_bcd  mnpo_broad_val4 + B oracle flags + C eval/multi-oracle +
                            copula derand K=5 + D PLS-DA folding (5 seeds)

    Resource optimization:
      - Dataset sharding uses empirical Val-4 runtime hints when available.
      - Falls back to tier-based runtime priors for datasets not present in partial Val-4 output.
      - Per-job max_workers defaults are tuned from Val-4 utilization:
        baseline=6, new_baseline_bcd=5 (up from Val-4's conservative global 2).
    """
    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, dataset_sets = _load_benchmark_registry()

    validation_ids = list(dataset_sets.get("validation_all") or [])
    if not validation_ids:
        raise RuntimeError("No validation_all datasets found in benchmark registry.")
    core_ids = set(dataset_sets.get("core") or [])
    extended_validation_ids = sorted(set(validation_ids) - core_ids)
    if not extended_validation_ids:
        raise RuntimeError(
            "Validation-5 expects the extended validation catalog, but no extended-only "
            "datasets were found in DATASET_SETS['validation_all']."
        )

    for required in ("mnpo_class_pareto_extended", "mnpo_broad_val4"):
        if required not in fs_method_sets:
            raise RuntimeError(f"Missing required method set {required!r} in FS_METHOD_SETS.")

    if val4_root is None:
        val4_root = REPO_ROOT / "run_artifacts" / "validation-4" / "val4_6pods_merged"
    val4_runs_csv = val4_root / "_aggregate" / "benchmark_df_fs_runs__all_jobs.csv"
    runtime_hints = _load_val4_runtime_hints(val4_runs_csv)
    manifest_meta = _load_hf_manifest_metadata()
    runtime_map = _estimate_val5_runtime_map(
        validation_ids=validation_ids,
        benchmark_datasets=benchmark_datasets,
        runtime_hints=runtime_hints,
        manifest_meta=manifest_meta,
    )

    # Partition by expected combined runtime across both profiles.
    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        base_t = float((runtime_map.get("baseline") or {}).get(ds_id, 1.0))
        bcd_t = float((runtime_map.get("new_baseline_bcd") or {}).get(ds_id, 1.0))
        # 5 seeds each profile.
        ds_items.append((str(ds_id), float((base_t + bcd_t) * len(VALIDATION_SEEDS))))
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_extra_args: Tuple[str, ...] = ("--emit-summary", "--compute-budget", "standard")
    common_job_params = dict(
        seeds=list(VALIDATION_SEEDS),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        task_timeout_sec=0,
        fs_method_timeout_sec=0,
    )

    _oracle_flags: Tuple[str, ...] = (
        "--enable-fs-mnpo-paradigm-aware-prior",
        "--fs-use-qre-smoothing",
        "--fs-use-oracle-redundancy-penalty",
        "--fs-compute-tremble-sensitivity",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance",
        "--screening-enabled",
        "--screening-method", "evalue",
    )

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            profile_id="baseline",
            fs_method_set="mnpo_class_pareto_extended",
            extra_args=tuple(),
            weight_mult=1.0,
            notes="Val-5 anchor baseline (production defaults).",
        ),
        BenchmarkProfile(
            profile_id="new_baseline_bcd",
            fs_method_set="mnpo_broad_val4",
            extra_args=(
                *_oracle_flags,
                "--eval-models-enabled",
                "--mnpo-performance-oracle-mode", "multi_model_oracles",
                "--eval-models", "lr_l2,linear_svc,rf_small",
                "--eval-aggregate", "mean",
                "--fs-copula-derandomize-runs", "5",
                "--folding-method", "pls_da",
            ),
            weight_mult=1.0,
            notes="Complete B+C+D stack candidate for production promotion.",
        ),
    ]

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError("Invalid validation5 profile(s):\n- " + "\n- ".join(invalid))

    workers_by_profile = {
        "baseline": 6,
        "new_baseline_bcd": 5,
    }

    jobs: List[Job] = []
    for prof in profiles:
        profile_runtime = dict(runtime_map.get(prof.profile_id) or {})
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_seed_sec = 0.0
            for ds_id in ds_list:
                if ds_id in profile_runtime:
                    part_seed_sec += float(profile_runtime[ds_id])
                else:
                    spec = benchmark_datasets.get(ds_id)
                    fallback = float(_dataset_weight(spec) * 220.0)
                    if prof.profile_id != "baseline":
                        fallback *= 1.5
                    part_seed_sec += fallback
            part_weight = float(part_seed_sec * len(VALIDATION_SEEDS))

            jobs.append(
                _job(
                    f"val5/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    max_workers=int(workers_by_profile.get(prof.profile_id, 4)),
                    **common_job_params,
                )
            )

    return jobs


def build_jobs_validation6(*, dataset_shards: int = 6, val5_root: Optional[Path] = None) -> List[Job]:
    """Validation-6: two-profile signal check on the 67-dataset Val-5 catalog.

    Profiles:
      A baseline : production default (mnpo_class_pareto_extended, PLS-DA, k=6)
      B full     : everything enabled — broad method stack, all oracles,
                   adaptive sizing, WSNR prefilter, PLS-DA, wmw_auc multiclass,
                   tighter diversity filter (portfolio sizing fixes from Phase 6B').

    Design rationale (first-principles, Feb 2026):
    - Only 2 profiles: paired comparison maximises statistical power.
    - Both profiles co-located on the same pods for valid pairing.
    - 67 datasets × 5 seeds = 335 runs per profile = 670 total runs.
    - Runtime-informed sharding from Val-5 empirical data.
    """
    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, dataset_sets = _load_benchmark_registry()

    validation_all = list(dataset_sets.get("validation_all") or [])
    validation_ids = [ds_id for ds_id in validation_all if not str(ds_id).startswith("rv_")]
    if not validation_ids:
        raise RuntimeError("No Val-6 datasets found after excluding results-validation (rv_*) datasets.")
    if len(validation_ids) != 67:
        raise RuntimeError(
            f"Val-6 expects 67 datasets (Val-5 catalog), found {len(validation_ids)} after rv_* exclusion."
        )

    for required in ("mnpo_class_pareto_extended", "mnpo_broad_val4"):
        if required not in fs_method_sets:
            raise RuntimeError(f"Missing required method set {required!r} in FS_METHOD_SETS.")

    if val5_root is None:
        val5_root = REPO_ROOT / "run_artifacts" / "validation-5" / "val5_6pods_merged"
    val5_runs_csv = val5_root / "_aggregate" / "benchmark_df_fs_runs__all_jobs.csv"
    runtime_hints = _load_val4_runtime_hints(val5_runs_csv)
    manifest_meta = _load_hf_manifest_metadata()

    baseline_hist = dict(runtime_hints.get("baseline") or {})
    bcd_hist = dict(runtime_hints.get("new_baseline_bcd") or {})

    baseline_est: Dict[str, float] = {}
    full_est: Dict[str, float] = {}

    for ds_id in validation_ids:
        base_t = float(baseline_hist.get(ds_id, 0.0))
        if base_t <= 0.0:
            spec = benchmark_datasets.get(ds_id)
            base_t = float(max(60.0, _dataset_weight(spec) * 220.0))
        baseline_est[ds_id] = float(base_t)

        # Full profile estimate: Val-5 BCD runtime × 1.20 (conservative overhead
        # for additional oracles, WSNR prefilter, adaptive sizing, wmw_auc multiclass).
        full_t = float(bcd_hist.get(ds_id, 0.0))
        if full_t <= 0.0:
            full_t = float(base_t * 1.80)
        else:
            full_t = float(full_t * 1.20)
        full_est[ds_id] = float(max(base_t * 1.10, full_t))

    # Partition datasets by combined expected runtime across both profiles.
    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        ds_items.append(
            (
                str(ds_id),
                float(
                    (baseline_est[ds_id] + full_est[ds_id])
                    * len(VALIDATION_SEEDS)
                ),
            )
        )
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_extra_args: Tuple[str, ...] = ("--emit-summary", "--compute-budget", "standard")
    common_job_params = dict(
        seeds=list(VALIDATION_SEEDS),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        task_timeout_sec=0,
        fs_method_timeout_sec=0,
    )

    # Full treatment flags: everything implemented in Phase 6A/6B/6B'.
    # Combines: oracle bundle (CVaR/Shapley/Complementarity/UBayFS),
    # PLS-DA overlay, WSNR prefilter union, adaptive portfolio sizing [4,8],
    # portfolio size guard, paradigm-aware prior, QRE smoothing, redundancy
    # penalty, mRMR MI, e-value screening, multi-model oracles, derandomized
    # knockoffs, and wmw_auc multiclass (via registry fix T-R-193).
    full_treatment_flags: Tuple[str, ...] = (
        # PLS-DA overlay (T-R-171): production default for >=5 classes
        "--folding-method", "pls_da",
        # Adaptive portfolio sizing with bounds [4,8] (T-R-190)
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        # Portfolio size guard (T-R-191)
        "--fs-portfolio-size-guard", "warn",
        # Paradigm-aware prior (T-R-128)
        "--enable-fs-mnpo-paradigm-aware-prior",
        # QRE smoothing + oracle redundancy penalty
        "--fs-use-qre-smoothing",
        "--fs-use-oracle-redundancy-penalty",
        # mRMR MI redundancy (T-R-126)
        "--enable-fs-mrmr-mi-redundancy",
        # Multi-strategy prefilter union (T-R-127) with WSNR (T-R-172)
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        # E-value screening (T-R-111)
        "--screening-enabled",
        "--screening-method", "evalue",
        # Multi-model performance oracles
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        # Derandomized knockoffs (T-R-110)
        "--fs-copula-derandomize-runs", "5",
        # CVaR oracle (T-R-181)
        "--fs-use-cvar-oracle",
        "--fs-cvar-alpha", "0.33",
        # Complementarity diversity mode (T-R-185)
        "--fs-diversity-oracle-mode", "complementarity",
        # Shapley weight calibration (T-R-184)
        "--fs-oracle-weighting-mode", "shapley",
        "--fs-shapley-n-coalitions-max", "4096",
        # UBayFS Bayesian ensemble (T-R-186)
        "--fs-use-ubayfs-oracle",
        "--fs-ubayfs-n-bootstrap", "32",
        "--fs-ubayfs-min-n", "100",
        "--fs-ubayfs-prior-weight", "0.0",
    )

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            profile_id="baseline",
            fs_method_set="mnpo_class_pareto_extended",
            extra_args=(
                # Production default: PLS-DA enabled (gated at >=5 classes).
                "--folding-method", "pls_da",
            ),
            notes="Val-6 control: production default (k=6, PLS-DA, 7-method stack).",
        ),
        BenchmarkProfile(
            profile_id="full",
            fs_method_set="mnpo_broad_val4",
            extra_args=full_treatment_flags,
            notes=(
                "Val-6 treatment: everything enabled — broad 16-method stack, "
                "all oracles (CVaR/Shapley/Complementarity/UBayFS), PLS-DA, WSNR, "
                "adaptive sizing [4,8], e-value screening, wmw_auc multiclass."
            ),
        ),
    ]

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError("Invalid validation6 profile(s):\n- " + "\n- ".join(invalid))

    workers_by_profile = {
        "baseline": 6,
        "full": 5,
    }

    runtime_by_profile = {
        "baseline": baseline_est,
        "full": full_est,
    }

    jobs: List[Job] = []
    for prof in profiles:
        profile_runtime = dict(runtime_by_profile.get(prof.profile_id) or {})
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_seed_sec = 0.0
            for ds_id in ds_list:
                part_seed_sec += float(profile_runtime.get(ds_id, 180.0))
            part_weight = float(part_seed_sec * len(VALIDATION_SEEDS))
            jobs.append(
                _job(
                    f"val6/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    max_workers=int(workers_by_profile.get(prof.profile_id, 4)),
                    **common_job_params,
                )
            )

    return jobs


def build_jobs_validation7(*, dataset_shards: int = 6, val6_root: Optional[Path] = None) -> List[Job]:
    """Validation-7: classifier upgrade + continuous shrinkage on the 67-dataset catalog.

    Profiles:
      A baseline  : production anchor (mnpo_class_pareto_extended, PLS-DA, k=6)
                    — unchanged from Val-6 baseline.
      B candidate : Val-6 full treatment + Bayesian shrinkage + variance-penalized
                    adaptive sizing.  Phase 4 oracles (TreeSHAP T-R-158, OAE-Net
                    T-R-161, Interaction T-R-142) are **opt-in disabled** per D-RM9-7.

    Design rationale (Phase 8, Feb 2026):
    - Only 2 profiles: paired comparison maximises statistical power.
    - Both profiles co-located on every pod for valid pairing.
    - 67 datasets × 5 seeds = 335 runs per profile = 670 total runs.
    - Runtime-informed sharding from Val-6 empirical data.
    - Candidate expected ≤ 2.0× baseline runtime (shrinkage reduces oracle overhead).
    """
    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, dataset_sets = _load_benchmark_registry()

    validation_all = list(dataset_sets.get("validation_all") or [])
    validation_ids = [ds_id for ds_id in validation_all if not str(ds_id).startswith("rv_")]
    if not validation_ids:
        raise RuntimeError("No Val-7 datasets found after excluding results-validation (rv_*) datasets.")
    if len(validation_ids) != 67:
        raise RuntimeError(
            f"Val-7 expects 67 datasets (extended catalog), found {len(validation_ids)} after rv_* exclusion."
        )

    for required in ("mnpo_class_pareto_extended", "mnpo_broad_val4"):
        if required not in fs_method_sets:
            raise RuntimeError(f"Missing required method set {required!r} in FS_METHOD_SETS.")

    # Load Val-6 runtime hints for sharding.
    if val6_root is None:
        val6_root = REPO_ROOT / "run_artifacts" / "validation-6" / "val6_finished_pull"
    val6_runtime = _load_val6_runtime_hints_from_summaries(val6_root)
    manifest_meta = _load_hf_manifest_metadata()

    baseline_hist = dict(val6_runtime.get("baseline") or {})
    full_hist = dict(val6_runtime.get("full") or {})

    baseline_est: Dict[str, float] = {}
    candidate_est: Dict[str, float] = {}

    for ds_id in validation_ids:
        # Baseline estimate: use Val-6 baseline runtime directly.
        base_t = float(baseline_hist.get(ds_id, 0.0))
        if base_t <= 0.0:
            spec = benchmark_datasets.get(ds_id)
            base_t = float(max(60.0, _dataset_weight(spec) * 220.0))
        baseline_est[ds_id] = float(base_t)

        # Candidate estimate: Val-6 full runtime × 0.95 (conservative: shrinkage
        # reduces oracle overhead, but classifier upgrade may add FLAML overhead
        # on medium/large datasets while being neutral on small-n datasets
        # where safety guards trigger sklearn fallback).
        full_t = float(full_hist.get(ds_id, 0.0))
        if full_t <= 0.0:
            full_t = float(base_t * 1.80)
        else:
            full_t = float(full_t * 0.95)
        candidate_est[ds_id] = float(max(base_t * 1.10, full_t))

    # Partition datasets by combined expected runtime across both profiles.
    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        ds_items.append(
            (
                str(ds_id),
                float(
                    (baseline_est[ds_id] + candidate_est[ds_id])
                    * len(VALIDATION_SEEDS)
                ),
            )
        )
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_extra_args: Tuple[str, ...] = ("--emit-summary", "--compute-budget", "standard")
    common_job_params = dict(
        seeds=list(VALIDATION_SEEDS),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        task_timeout_sec=0,
        fs_method_timeout_sec=0,
    )

    # Candidate treatment flags: Val-6 full + T-R-179 shrinkage + classifier upgrade.
    # Phase 4 oracles (TreeSHAP T-R-158, OAE-Net T-R-161, Interaction T-R-142)
    # are NOT included (opt-in disabled per D-RM9-7).
    candidate_treatment_flags: Tuple[str, ...] = (
        # PLS-DA overlay (T-R-171)
        "--folding-method", "pls_da",
        # Adaptive portfolio sizing with bounds [4,8] (T-R-190)
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        # Portfolio size guard (T-R-191)
        "--fs-portfolio-size-guard", "warn",
        # Paradigm-aware prior (T-R-128)
        "--enable-fs-mnpo-paradigm-aware-prior",
        # QRE smoothing + oracle redundancy penalty
        "--fs-use-qre-smoothing",
        "--fs-use-oracle-redundancy-penalty",
        # mRMR MI redundancy (T-R-126)
        "--enable-fs-mrmr-mi-redundancy",
        # Multi-strategy prefilter union (T-R-127) with WSNR (T-R-172)
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        # E-value screening (T-R-111)
        "--screening-enabled",
        "--screening-method", "evalue",
        # Multi-model performance oracles
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        # Derandomized knockoffs (T-R-110)
        "--fs-copula-derandomize-runs", "5",
        # CVaR oracle (T-R-181)
        "--fs-use-cvar-oracle",
        "--fs-cvar-alpha", "0.33",
        # Complementarity diversity mode (T-R-185)
        "--fs-diversity-oracle-mode", "complementarity",
        # Shapley weight calibration (T-R-184)
        "--fs-oracle-weighting-mode", "shapley",
        "--fs-shapley-n-coalitions-max", "4096",
        # UBayFS Bayesian ensemble (T-R-186)
        "--fs-use-ubayfs-oracle",
        "--fs-ubayfs-n-bootstrap", "32",
        "--fs-ubayfs-min-n", "100",
        "--fs-ubayfs-prior-weight", "0.0",
        # NEW in Val-7: Bayesian shrinkage for Shapley oracle (T-R-179)
        "--shapley-bayesian-shrinkage",
        # NEW in Val-7: variance-penalized adaptive sizing (T-R-179)
        "--adaptive-sizing-variance-penalty",
    )

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            profile_id="baseline",
            fs_method_set="mnpo_class_pareto_extended",
            extra_args=(
                "--folding-method", "pls_da",
            ),
            notes="Val-7 control: production anchor (k=6, PLS-DA, 7-method stack). Unchanged from Val-6 baseline.",
        ),
        BenchmarkProfile(
            profile_id="candidate",
            fs_method_set="mnpo_broad_val4",
            extra_args=candidate_treatment_flags,
            notes=(
                "Val-7 treatment: Val-6 full + Bayesian shrinkage (T-R-179) + "
                "variance-penalized adaptive sizing (T-R-179). "
                "Phase 4 oracles (TreeSHAP, OAE-Net, Interaction) opt-in DISABLED (D-RM9-7)."
            ),
        ),
    ]

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError("Invalid validation7 profile(s):\n- " + "\n- ".join(invalid))

    workers_by_profile = {
        "baseline": 6,
        "candidate": 5,
    }

    runtime_by_profile = {
        "baseline": baseline_est,
        "candidate": candidate_est,
    }

    jobs: List[Job] = []
    for prof in profiles:
        profile_runtime = dict(runtime_by_profile.get(prof.profile_id) or {})
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_seed_sec = 0.0
            for ds_id in ds_list:
                part_seed_sec += float(profile_runtime.get(ds_id, 180.0))
            part_weight = float(part_seed_sec * len(VALIDATION_SEEDS))
            jobs.append(
                _job(
                    f"val7/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    max_workers=int(workers_by_profile.get(prof.profile_id, 4)),
                    **common_job_params,
                )
            )

    return jobs


def build_jobs_validation8(*, dataset_shards: int = 6, val7_root: Optional[Path] = None) -> List[Job]:
    """Validation-8: all approved fixes + Stage-2 classifier support on 67 datasets.

    Profiles:
      A baseline  : production anchor (mnpo_class_pareto_extended, PLS-DA, k=6).
      B candidate : Val-7 candidate stack + RNA-seq NB-LRT prefilter + Stage-2
                    classifier support (FLAML backend with expanded fallback
                    model candidate set).

    This keeps paired comparison statistical power while incorporating the
    RNA-seq prefilter fixes (T-R-176/T-R-178) and final-stage classifier
    upgrade support (T-R-200..204/206/209/210/212).
    """
    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, dataset_sets = _load_benchmark_registry()

    validation_all = list(dataset_sets.get("validation_all") or [])
    validation_ids = [ds_id for ds_id in validation_all if not str(ds_id).startswith("rv_")]
    if not validation_ids:
        raise RuntimeError("No Val-8 datasets found after excluding results-validation (rv_*) datasets.")
    if len(validation_ids) != 67:
        raise RuntimeError(
            f"Val-8 expects 67 datasets (extended catalog), found {len(validation_ids)} after rv_* exclusion."
        )

    for required in ("mnpo_class_pareto_extended", "mnpo_broad_val4"):
        if required not in fs_method_sets:
            raise RuntimeError(f"Missing required method set {required!r} in FS_METHOD_SETS.")

    # Load Val-7 runtime hints for sharding.
    if val7_root is None:
        val7_root = REPO_ROOT / "run_artifacts" / "validation-7" / "val7_6pods_merged"
    val7_runtime = _load_val7_runtime_hints_from_summaries(val7_root)
    _ = _load_hf_manifest_metadata()

    baseline_hist = dict(val7_runtime.get("baseline") or {})
    val7_candidate_hist = dict(val7_runtime.get("candidate") or {})

    baseline_est: Dict[str, float] = {}
    candidate_est: Dict[str, float] = {}

    for ds_id in validation_ids:
        base_t = float(baseline_hist.get(ds_id, 0.0))
        if base_t <= 0.0:
            spec = benchmark_datasets.get(ds_id)
            base_t = float(max(60.0, _dataset_weight(spec) * 220.0))
        baseline_est[ds_id] = float(base_t)

        # Val-8 candidate adds NB-LRT + Stage-2 classifier support, so use a
        # conservative uplift over Val-7 candidate where available.
        cand_t = float(val7_candidate_hist.get(ds_id, 0.0))
        if cand_t <= 0.0:
            cand_t = float(base_t * 1.90)
        else:
            cand_t = float(cand_t * 1.20)
        candidate_est[ds_id] = float(max(base_t * 1.25, cand_t))

    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        ds_items.append(
            (
                str(ds_id),
                float(
                    (baseline_est[ds_id] + candidate_est[ds_id])
                    * len(VALIDATION_SEEDS)
                ),
            )
        )
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_extra_args: Tuple[str, ...] = ("--emit-summary", "--compute-budget", "standard")
    common_job_params = dict(
        seeds=list(VALIDATION_SEEDS),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        task_timeout_sec=0,
        fs_method_timeout_sec=0,
    )

    candidate_treatment_flags: Tuple[str, ...] = (
        # Preserve Val-7 candidate controls.
        "--folding-method", "pls_da",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-fs-mnpo-paradigm-aware-prior",
        "--fs-use-qre-smoothing",
        "--fs-use-oracle-redundancy-penalty",
        "--fs-compute-tremble-sensitivity",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--fs-copula-derandomize-runs", "5",
        "--fs-use-cvar-oracle",
        "--fs-cvar-alpha", "0.33",
        "--fs-diversity-oracle-mode", "complementarity",
        "--fs-oracle-weighting-mode", "shapley",
        "--fs-shapley-n-coalitions-max", "4096",
        "--fs-use-ubayfs-oracle",
        "--fs-ubayfs-n-bootstrap", "32",
        "--fs-ubayfs-min-n", "100",
        "--fs-ubayfs-prior-weight", "0.0",
        "--shapley-bayesian-shrinkage",
        "--adaptive-sizing-variance-penalty",
        # NEW in Val-8: RNA-seq domain-aware NB-LRT prefilter signal.
        "--enable-prefilter-rnaseq-nb-lrt",
        "--prefilter-rnaseq-nb-lrt-alpha", "0.10",
        # NEW in Val-8: final-stage classifier backend support.
        "--classification-backend", "flaml",
        "--flaml-time-budget", "90",
        # Expanded sklearn fallback candidate pool (used directly when FLAML
        # is unsupported on a dataset or unavailable on a worker).
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "shrinkage_lda",
        "nsc", "pls_da_classifier", "gpc", "nb", "vote_ensemble",
        "elastic_net_lr", "rf", "knn",
        # Contain fallback CV breadth in pathological regimes.
        "--enable-model-cv-runtime-containment",
        "--model-cv-runtime-max-candidates", "8",
    )

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            profile_id="baseline",
            fs_method_set="mnpo_class_pareto_extended",
            extra_args=(
                "--folding-method", "pls_da",
            ),
            notes=(
                "Val-8 control: production anchor (k=6, PLS-DA, 7-method stack). "
                "Unchanged from Val-7 baseline."
            ),
        ),
        BenchmarkProfile(
            profile_id="candidate",
            fs_method_set="mnpo_broad_val4",
            extra_args=candidate_treatment_flags,
            notes=(
                "Val-8 treatment: Val-7 candidate + RNA-seq NB-LRT prefilter "
                "(T-R-178) + Stage-2 classifier support (FLAML backend + expanded "
                "fallback candidates including NSC/PLS-DA-classifier/Shrinkage-LDA/GPC)."
            ),
        ),
    ]

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError("Invalid validation8 profile(s):\n- " + "\n- ".join(invalid))

    workers_by_profile = {
        "baseline": 6,
        "candidate": 4,
    }

    runtime_by_profile = {
        "baseline": baseline_est,
        "candidate": candidate_est,
    }

    jobs: List[Job] = []
    for prof in profiles:
        profile_runtime = dict(runtime_by_profile.get(prof.profile_id) or {})
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_seed_sec = 0.0
            for ds_id in ds_list:
                part_seed_sec += float(profile_runtime.get(ds_id, 180.0))
            part_weight = float(part_seed_sec * len(VALIDATION_SEEDS))
            jobs.append(
                _job(
                    f"val8/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    max_workers=int(workers_by_profile.get(prof.profile_id, 4)),
                    **common_job_params,
                )
            )

    return jobs


def build_jobs_validation9(
    *,
    dataset_shards: int = 6,
    val8_root: Optional[Path] = None,
    runtime_profile: str = "full",
) -> List[Job]:
    """Validation-9: comprehensive full-stack run with classifier selection A/B.

    Profiles:
      A legacy_full : full feature stack with legacy classifier selection.
      B mnpo_hybrid : same stack + MNPO-hybrid classifier selection mode.

    Design goals:
    - Include all implemented pre-Val-9 features in both profiles.
    - Keep a paired comparison where the primary contrast is classifier selection
      mode (`legacy` vs `mnpo_hybrid`).
    - Use Val-8 runtime hints for shard balancing.
    - Support a runtime-tuned profile that preserves method/oracle coverage while
      reducing expensive inner-loop budgets.
    """
    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, dataset_sets = _load_benchmark_registry()

    validation_all = list(dataset_sets.get("validation_all") or [])
    validation_ids = [ds_id for ds_id in validation_all if not str(ds_id).startswith("rv_")]
    if not validation_ids:
        raise RuntimeError("No Val-9 datasets found after excluding results-validation (rv_*) datasets.")
    if len(validation_ids) != 67:
        raise RuntimeError(
            f"Val-9 expects 67 datasets (extended catalog), found {len(validation_ids)} after rv_* exclusion."
        )
    runtime_profile_norm = str(runtime_profile or "full").strip().lower()
    if runtime_profile_norm not in {"full", "tuned"}:
        raise RuntimeError(
            f"Invalid validation9 runtime_profile={runtime_profile!r}; expected 'full' or 'tuned'."
        )

    if "mnpo_broad_all" not in fs_method_sets:
        raise RuntimeError("Missing required method set 'mnpo_broad_all' in FS_METHOD_SETS.")

    if val8_root is None:
        val8_root = REPO_ROOT / "run_artifacts" / "validation-8" / "val8_6pods_merged"
    val8_runtime = _load_val8_runtime_hints_from_summaries(val8_root)

    val8_baseline_hist = dict(val8_runtime.get("baseline") or {})
    val8_candidate_hist = dict(val8_runtime.get("candidate") or {})

    legacy_est: Dict[str, float] = {}
    hybrid_est: Dict[str, float] = {}
    for ds_id in validation_ids:
        # Start from the richer Val-8 candidate estimate when available since
        # Val-9 explicitly runs a comprehensive stack in both profiles.
        ref_t = float(val8_candidate_hist.get(ds_id, 0.0))
        if ref_t <= 0.0:
            base_t = float(val8_baseline_hist.get(ds_id, 0.0))
            if base_t <= 0.0:
                spec = benchmark_datasets.get(ds_id)
                base_t = float(max(60.0, _dataset_weight(spec) * 250.0))
            ref_t = float(base_t * 1.35)
        legacy_est[ds_id] = float(max(60.0, ref_t))
        # MNPO hybrid classifier mode adds oracle+per-family-HPO overhead.
        hybrid_est[ds_id] = float(max(legacy_est[ds_id] * 1.10, legacy_est[ds_id] * 1.22))

    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        ds_items.append(
            (
                str(ds_id),
                float((legacy_est[ds_id] + hybrid_est[ds_id]) * len(VALIDATION_SEEDS)),
            )
        )
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_extra_args: Tuple[str, ...] = ("--emit-summary", "--compute-budget", "standard")
    common_job_params = dict(
        seeds=list(VALIDATION_SEEDS),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        task_timeout_sec=0,
        fs_method_timeout_sec=0,
    )

    # Runtime-tuned profile keeps feature/method/oracle coverage intact while
    # reducing high-cost budget knobs to improve throughput on CPU-bound pods.
    if runtime_profile_norm == "tuned":
        common_job_params["task_timeout_sec"] = 10800
        common_job_params["fs_method_timeout_sec"] = 1800
        copula_derandomize_runs = "3"
        interaction_oracle_pair_cap = "12000"
        shapley_n_coalitions_max = "2048"
        ubayfs_n_bootstrap = "20"
        flaml_time_budget = "90"
        model_cv_runtime_max_candidates = "8"
        classifier_oracle_bbc_bootstrap_rounds = "120"
        workers_by_profile = {
            "legacy_full": 3,
            "mnpo_hybrid": 2,
        }
    else:
        copula_derandomize_runs = "5"
        interaction_oracle_pair_cap = "20000"
        shapley_n_coalitions_max = "4096"
        ubayfs_n_bootstrap = "32"
        flaml_time_budget = "120"
        model_cv_runtime_max_candidates = "10"
        classifier_oracle_bbc_bootstrap_rounds = "200"
        workers_by_profile = {
            "legacy_full": 4,
            "mnpo_hybrid": 3,
        }

    full_stack_flags: Tuple[str, ...] = (
        "--dist-criterion", "mnpo_oracle",
        "--df-lmoment-prescreen",
        "--df-lmoment-prescreen-max-candidates", "12",
        "--df-compute-crps",
        "--df-crps-uq-decomposition",
        "--df-mnpo-include-crps",
        "--df-mnpo-include-preq",
        "--df-use-qre-smoothing",
        "--df-use-oracle-redundancy-penalty",
        "--df-compute-tremble-sensitivity",
        "--folding-method", "pls_da",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-fs-mnpo-paradigm-aware-prior",
        "--fs-use-qre-smoothing",
        "--fs-use-oracle-redundancy-penalty",
        "--fs-compute-tremble-sensitivity",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--fs-copula-derandomize-runs", copula_derandomize_runs,
        "--fs-use-cvar-oracle",
        "--fs-cvar-alpha", "0.33",
        "--fs-use-interaction-oracle",
        "--fs-interaction-oracle-min-n-train", "150",
        "--fs-interaction-oracle-pool-size-cap", "64",
        "--fs-interaction-oracle-pair-cap", interaction_oracle_pair_cap,
        "--fs-diversity-oracle-mode", "complementarity",
        "--fs-oracle-weighting-mode", "shapley",
        "--fs-shapley-n-coalitions-max", shapley_n_coalitions_max,
        "--shapley-bayesian-shrinkage",
        "--adaptive-sizing-variance-penalty",
        "--fs-use-ubayfs-oracle",
        "--fs-ubayfs-n-bootstrap", ubayfs_n_bootstrap,
        "--fs-ubayfs-min-n", "100",
        "--fs-ubayfs-prior-weight", "0.0",
        "--fs-use-conformal-uq",
        "--fs-conformal-uq-alpha", "0.10",
        "--fs-conformal-uq-min-folds", "5",
        "--enable-prefilter-rnaseq-nb-lrt",
        "--prefilter-rnaseq-nb-lrt-alpha", "0.10",
        "--classification-backend", "flaml",
        "--flaml-time-budget", flaml_time_budget,
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "shrinkage_lda",
        "nsc", "pls_da_classifier", "gpc", "nb", "vote_ensemble",
        "elastic_net_lr", "rf", "knn", "xgb", "lgbm", "extra_tree", "catboost",
        "--include-nsc-model",
        "--include-pls-da-model",
        "--include-gpc-model",
        "--include-lgbm-model",
        "--include-extra-tree-model",
        "--include-catboost-model",
        "--enable-model-cv-runtime-containment",
        "--model-cv-runtime-max-candidates", model_cv_runtime_max_candidates,
        "--enable-fs-runtime-racing",
        "--fs-runtime-racing-mode", "successive_halving",
        "--fs-runtime-racing-stages", "2",
        "--fs-runtime-racing-confidence-bound", "hoeffding",
        "--fs-runtime-racing-delta", "0.10",
        "--enable-fs-rashomon",
        "--enable-ratio-features",
        "--ratio-selection-method", "ktsp",
        "--max-ratio-features", "30",
        "--enable-stage2-ratio-augmentation",
        "--stage2-ratio-max-features", "16",
        "--stage2-ratio-selection-method", "correlation",
        "--enable-maqc-pairing",
        "--maqc-fs-method-sets", "strict_plus_mrmr", "mnpo_rankagg_extended", "mnpo_ova_extended",
    )

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            profile_id="legacy_full",
            fs_method_set="mnpo_broad_all",
            extra_args=(
                *full_stack_flags,
                "--classifier-selection-mode", "legacy",
            ),
            notes=(
                "Val-9 control: full-stack feature set with legacy classifier selection "
                "(keeps comprehensive pipeline while isolating classifier-selection mode)."
            ),
        ),
        BenchmarkProfile(
            profile_id="mnpo_hybrid",
            fs_method_set="mnpo_broad_all",
            extra_args=(
                *full_stack_flags,
                "--classifier-selection-mode", "mnpo_hybrid",
                "--classifier-oracle-k", "2",
                "--classifier-oracle-weighting-mode", "tritrust",
                "--classifier-oracle-bbc-bootstrap-rounds", classifier_oracle_bbc_bootstrap_rounds,
                "--classifier-oracle-bbc-ci-level", "0.90",
                "--enable-classifier-oracle-ensemble",
            ),
            notes=(
                "Val-9 treatment: same full stack + MNPO hybrid classifier selection "
                "(regime gating, classifier oracles, per-family HPO)."
            ),
        ),
    ]

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError("Invalid validation9 profile(s):\n- " + "\n- ".join(invalid))

    runtime_by_profile = {
        "legacy_full": legacy_est,
        "mnpo_hybrid": hybrid_est,
    }

    jobs: List[Job] = []
    for prof in profiles:
        profile_runtime = dict(runtime_by_profile.get(prof.profile_id) or {})
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_seed_sec = 0.0
            for ds_id in ds_list:
                part_seed_sec += float(profile_runtime.get(ds_id, 180.0))
            part_weight = float(part_seed_sec * len(VALIDATION_SEEDS))
            jobs.append(
                _job(
                    f"val9/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    max_workers=int(workers_by_profile.get(prof.profile_id, 3)),
                    **common_job_params,
                )
            )
    return jobs


def build_jobs_validation10(
    *,
    dataset_shards: int = 6,
    val9_root: Optional[Path] = None,
) -> List[Job]:
    """Validation-10: simple-vs-MNPO usage across all pipeline stages.

    Profiles:
      A simple_all_stages:
        - DF criterion: simple
        - FS stack: strict_plus_mrmr
        - Classifier selection: legacy
      B mnpo_all_stages:
        - DF criterion: mnpo_oracle (+ DF oracle bundle)
        - FS stack: mnpo_broad_all (+ full MNPO oracle bundle)
        - Classifier selection: mnpo_hybrid

    Val-10 targets the extended validation catalog with HF as the authoritative
    source: datasets are restricted to ``validation_all ∩ HF_manifest_ids``.
    """
    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, dataset_sets = _load_benchmark_registry()

    validation_all_ids = list(dataset_sets.get("validation_all") or [])
    if not validation_all_ids:
        raise RuntimeError("No validation_all datasets found in benchmark registry.")

    hf_manifest_ids = set(_load_hf_manifest_metadata().keys())
    if not hf_manifest_ids:
        raise RuntimeError("HF bundle manifest metadata is empty; cannot build validation10 plan.")

    validation_ids = [str(ds_id) for ds_id in validation_all_ids if str(ds_id) in hf_manifest_ids]
    if not validation_ids:
        raise RuntimeError(
            "No validation_all datasets are available in HF bundle manifests. "
            "Set TABNETICS_HF_MANIFEST correctly and/or refresh train_data manifests."
        )

    missing_hf = [str(ds_id) for ds_id in validation_all_ids if str(ds_id) not in hf_manifest_ids]
    if missing_hf:
        preview = ", ".join(missing_hf[:10])
        suffix = " ..." if len(missing_hf) > 10 else ""
        print(
            f"[validation10] Skipping {len(missing_hf)} validation_all dataset(s) not present in "
            f"HF manifests: {preview}{suffix}",
            file=sys.stderr,
        )

    rv_ids = [ds_id for ds_id in validation_ids if str(ds_id).startswith("rv_")]
    if not rv_ids:
        raise RuntimeError(
            "Val-10 expects at least one rv_* dataset from the extended catalog, "
            "but HF-supported validation IDs contain none. Update HF manifests/bundle first."
        )

    for required in ("strict_plus_mrmr", "mnpo_broad_all"):
        if required not in fs_method_sets:
            raise RuntimeError(f"Missing required method set {required!r} in FS_METHOD_SETS.")

    if val9_root is None:
        val9_root = (
            REPO_ROOT
            / "run_artifacts"
            / "validation-9"
            / "val9_6pods_live_pull_20260223_165714"
        )
    val9_runtime = _load_val9_runtime_hints_from_summaries(val9_root)
    val9_legacy_hist = dict(val9_runtime.get("legacy_full") or {})
    val9_hybrid_hist = dict(val9_runtime.get("mnpo_hybrid") or {})

    simple_est: Dict[str, float] = {}
    mnpo_est: Dict[str, float] = {}
    for ds_id in validation_ids:
        base_t = float(val9_legacy_hist.get(ds_id, 0.0))
        if base_t <= 0.0:
            spec = benchmark_datasets.get(ds_id)
            base_t = float(max(60.0, _dataset_weight(spec) * 220.0))
        simple_est[ds_id] = float(base_t)

        mnpo_t = float(val9_hybrid_hist.get(ds_id, 0.0))
        if mnpo_t <= 0.0:
            mnpo_t = float(max(base_t * 1.65, base_t + 90.0))
        mnpo_est[ds_id] = float(max(base_t * 1.20, mnpo_t))

    # Partition by combined expected runtime across both profiles.
    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        ds_items.append(
            (
                str(ds_id),
                float((simple_est[ds_id] + mnpo_est[ds_id]) * len(VALIDATION_SEEDS)),
            )
        )
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_extra_args: Tuple[str, ...] = ("--emit-summary", "--compute-budget", "standard")
    shared_stage_flags: Tuple[str, ...] = (
        "--df-family-set", "flex",
        "--df-compute-ad",
        "--df-compute-qq-pp",
        "--df-compute-dip",
        "--df-interval-likelihood",
        "--df-compute-crps",
        "--df-crps-uq-decomposition",
        "--df-lmoment-prescreen",
        "--df-lmoment-prescreen-max-candidates", "12",
        "--folding-method", "pls_da",
        "--enable-prefilter-rnaseq-nb-lrt",
        "--prefilter-rnaseq-nb-lrt-alpha", "0.10",
        "--enable-classifier-conformal",
        "--classifier-conformal-alpha", "0.10",
        "--classifier-conformal-calibration-fraction", "0.25",
        "--classifier-conformal-min-calibration", "20",
        "--enable-stage2-ratio-augmentation",
        "--stage2-ratio-max-features", "16",
        "--stage2-ratio-selection-method", "correlation",
        "--enable-model-cv-runtime-containment",
    )
    common_job_params = dict(
        seeds=list(VALIDATION_SEEDS),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        progress_stall_watchdog_sec=1800,
        task_timeout_sec=21600,
        fs_method_timeout_sec=3600,
    )

    simple_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--mnpo-performance-oracle-mode", "single",
        "--fs-oracle-weighting-mode", "uniform",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
        "--model-cv-runtime-max-candidates", "8",
    )

    mnpo_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "mnpo_oracle",
        "--df-mnpo-include-crps",
        "--df-mnpo-include-preq",
        "--df-use-qre-smoothing",
        "--df-use-oracle-redundancy-penalty",
        "--df-compute-tremble-sensitivity",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-fs-mnpo-paradigm-aware-prior",
        "--fs-use-qre-smoothing",
        "--fs-use-oracle-redundancy-penalty",
        "--fs-compute-tremble-sensitivity",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--fs-copula-derandomize-runs", "3",
        "--fs-use-cvar-oracle",
        "--fs-cvar-alpha", "0.33",
        "--fs-use-interaction-oracle",
        "--fs-interaction-oracle-min-n-train", "150",
        "--fs-interaction-oracle-pool-size-cap", "64",
        "--fs-interaction-oracle-pair-cap", "12000",
        "--fs-diversity-oracle-mode", "complementarity",
        "--fs-oracle-weighting-mode", "shapley",
        "--fs-shapley-n-coalitions-max", "2048",
        "--shapley-bayesian-shrinkage",
        "--adaptive-sizing-variance-penalty",
        "--fs-use-ubayfs-oracle",
        "--fs-ubayfs-n-bootstrap", "20",
        "--fs-ubayfs-min-n", "100",
        "--fs-ubayfs-prior-weight", "0.0",
        "--fs-use-conformal-uq",
        "--fs-conformal-uq-alpha", "0.10",
        "--fs-conformal-uq-min-folds", "5",
        "--enable-fs-runtime-racing",
        "--fs-runtime-racing-mode", "successive_halving",
        "--fs-runtime-racing-stages", "2",
        "--fs-runtime-racing-confidence-bound", "hoeffding",
        "--fs-runtime-racing-delta", "0.10",
        "--enable-fs-rashomon",
        "--fs-rashomon-max-models", "12",
        "--fs-rashomon-score-tolerance", "0.02",
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--classification-backend", "flaml",
        "--flaml-time-budget", "120",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "shrinkage_lda",
        "nsc", "pls_da_classifier", "gpc", "nb", "vote_ensemble",
        "elastic_net_lr", "rf", "knn", "xgb", "lgbm", "extra_tree", "catboost",
        "--include-nsc-model",
        "--include-pls-da-model",
        "--include-gpc-model",
        "--include-lgbm-model",
        "--include-extra-tree-model",
        "--include-catboost-model",
        "--model-cv-runtime-max-candidates", "10",
    )

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            profile_id="simple_all_stages",
            fs_method_set="strict_plus_mrmr",
            extra_args=simple_flags,
            notes=(
                "Val-10 control: simple scoring/selection path in DF + FS + classifier "
                "(dist_criterion=simple, strict_plus_mrmr, classifier legacy)."
            ),
        ),
        BenchmarkProfile(
            profile_id="mnpo_all_stages",
            fs_method_set="mnpo_broad_all",
            extra_args=mnpo_flags,
            notes=(
                "Val-10 treatment: MNPO usage across all stages (DF MNPO oracle, "
                "FS MNPO full stack, classifier mnpo_hybrid)."
            ),
        ),
    ]

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError("Invalid validation10 profile(s):\n- " + "\n- ".join(invalid))

    runtime_by_profile = {
        "simple_all_stages": simple_est,
        "mnpo_all_stages": mnpo_est,
    }

    jobs: List[Job] = []
    for prof in profiles:
        profile_runtime = dict(runtime_by_profile.get(prof.profile_id) or {})
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_seed_sec = 0.0
            for ds_id in ds_list:
                part_seed_sec += float(profile_runtime.get(ds_id, 180.0))
            part_weight = float(part_seed_sec * len(VALIDATION_SEEDS))
            jobs.append(
                _job(
                    f"val10/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    **common_job_params,
                )
            )
    return jobs


def _load_val10_runtime_hints_from_summaries(val10_root: Path) -> Dict[str, Dict[str, float]]:
    return _load_runtime_hints_from_summaries(val10_root, phase_tag="val10")


def build_jobs_validation11(
    *,
    dataset_shards: int = 4,
    val10_root: Optional[Path] = None,
) -> List[Job]:
    """Validation-11 v2: 2×2 factorial stage decomposition + extensions.

    Val-11 tests whether MNPO's value comes from Stage 2 (game-theoretic FS) or
    Stage 3 (ensemble classification) via a 2×2 factorial design, plus three
    extension profiles probing CVaR tail-risk, experimental FS features, and
    alternative screening/knockoff methods:

        |                | Simple Classifier   | MNPO Ensemble Classifier |
        |----------------|---------------------|--------------------------|
        | Simple FS      | A simple_control    | C simple_fs_mnpo_clf     |
        | MNPO FS        | D mnpo_fs_simple_clf| B mnpo_improved          |

    Extensions:
      E mnpo_improved_cvar:            B + CVaR tail-risk oracle (alpha=0.33)
      F mnpo_improved_experimental_fs: B + 11 experimental FS features
      G mnpo_improved_alt_screening:   B + STIR, MPS, ratio features, DeepDRK

    Profiles (7 total):
      A simple_control:               simple dist + strict_plus_mrmr FS + sklearn legacy clf
      B mnpo_improved:                mnpo_oracle dist + mnpo_broad_all FS (banzhaf) + flaml mnpo_hybrid clf
      C simple_fs_mnpo_clf:           simple dist + strict_plus_mrmr FS + flaml mnpo_hybrid clf
      D mnpo_fs_simple_clf:           simple dist + mnpo_broad_all FS (banzhaf) + sklearn legacy clf
      E mnpo_improved_cvar:           B + CVaR oracle (alpha=0.33)
      F mnpo_improved_experimental_fs: B + wrapper-refine, stability-loss, adaptive-imbalance,
                                        runtime-racing, CPSS, class-Pareto, OVA, NSC-deep,
                                        Borda, copula-stabilizer, redundancy-filter
      G mnpo_improved_alt_screening:  B + STIR screening, MPS estimator, ratio features, DeepDRK

    Uses a focused selection of 24 datasets (8 MNPO-wins, 8 simple-wins,
    4 regression ties, 4 tier coverage) for <6h wall time across 4 hosts.

    Statistical contrasts (21 pairwise, Bonferroni α ≈ 0.0024):
      - Classifier main effect: (B+C)/2 − (A+D)/2
      - FS main effect: (B+D)/2 − (A+C)/2
      - FS×Classifier interaction: B − C − D + A
      - CVaR tail-risk effect: E − B
      - Experimental FS effect: F − B
      - Alt screening effect: G − B
    """
    # --- Focused 24-dataset selection for Val-11 v2 ---
    # Selected from Val-10 paired results to maximize information density
    # while keeping runtime within 6h across 4 lab hosts.
    VAL11_FOCUSED_DATASETS: List[str] = [
        # MNPO-wins (8): datasets where MNPO outperformed simple in Val-10
        "dlbcl_shipp",               # Δ=−0.100, 2-class easy, n/K=38.5
        "cumida_brain_gse50161",     # Δ=−0.100, 4-class hard, n/K=27
        "leukemia_1_72_3class",      # Δ=−0.067, 3-class medium, n/K=24
        "cll_sub_111",               # Δ=−0.057, 3-class medium, n/K=37
        "carcinom_11class",          # Δ=−0.050, 11-class hard, n/K=15.8
        "xena_tcga_gbm",            # Δ=−0.067, 4-class hard, n/K=41
        "gli_85",                    # Δ=−0.045, 2-class medium, n/K=42.5
        "glioma_50_4class",          # Δ=−0.040, 4-class medium, n/K=12.5
        # Simple-wins (8): datasets where simple outperformed MNPO in Val-10
        "hf_breast_ge_mubashir1837", # Δ=+0.167, 2-class hard, n/K=25.5
        "cns_pomeroy",               # Δ=+0.163, 2-class medium, n/K=30
        "cumida_colorectal_gse44861",# Δ=+0.157, 2-class medium, n/K=55.5
        "madelon_nips03",            # Δ=+0.120, 2-class medium, n/K=1300
        "xena_tcga_coad_cms",        # Δ=+0.113, 2-class hard, n/K=161.5
        "nci9_60_9class",            # Δ=+0.077, 9-class very_hard, n/K=6.7
        "tox_171",                   # Δ=+0.032, 4-class medium, n/K=42.8
        "colon_alon",                # Δ=+0.035, 2-class easy, n/K=31
        # Regression ties (4): both pipelines tied in Val-10
        "srbct_khan",                # Δ=0.000, 4-class easy, ceiling BA=1.0
        "ovarian_petricoin",         # Δ=0.000, 2-class easy, near-ceiling
        "prostate_singh",            # Δ=−0.020, 2-class easy, close tie
        "lymphoma_3",                # Δ=−0.013, 3-class easy, multiclass tie
        # Tier coverage (4): fill representation gaps
        "leukemia_golub",            # 2-class easy, classic benchmark
        "arcene_nips03",             # 2-class medium, NIPS benchmark
        "nci60_strict_holdout",      # 9-class very_hard, extreme challenge
        "cumida_leukemia_subtypes",  # 7-class hard, multiclass hard
    ]

    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, dataset_sets = _load_benchmark_registry()

    hf_manifest_ids = set(_load_hf_manifest_metadata().keys())
    if not hf_manifest_ids:
        raise RuntimeError("HF bundle manifest metadata is empty; cannot build validation11 plan.")

    # Use focused dataset list, filtered to available HF manifests
    validation_ids = [ds_id for ds_id in VAL11_FOCUSED_DATASETS if ds_id in hf_manifest_ids]
    if not validation_ids:
        raise RuntimeError("No Val-11 focused datasets are available in HF bundle manifests.")

    missing_hf = [ds_id for ds_id in VAL11_FOCUSED_DATASETS if ds_id not in hf_manifest_ids]
    if missing_hf:
        preview = ", ".join(missing_hf[:10])
        suffix = " ..." if len(missing_hf) > 10 else ""
        print(
            f"[validation11] Skipping {len(missing_hf)} focused dataset(s) not present in "
            f"HF manifests: {preview}{suffix}",
            file=sys.stderr,
        )

    for required in ("strict_plus_mrmr", "mnpo_broad_all"):
        if required not in fs_method_sets:
            raise RuntimeError(f"Missing required method set {required!r} in FS_METHOD_SETS.")

    # --- Runtime estimation from Val-10 history ---
    if val10_root is None:
        val10_root = (
            REPO_ROOT
            / "run_artifacts"
            / "validation-10"
            / "val10_finished_done_only_pull_20260227_111010"
        )
    val10_runtime = _load_val10_runtime_hints_from_summaries(val10_root)
    val10_simple_hist = dict(val10_runtime.get("simple_all_stages") or {})
    val10_mnpo_hist = dict(val10_runtime.get("mnpo_all_stages") or {})

    simple_est: Dict[str, float] = {}
    mnpo_est: Dict[str, float] = {}
    mnpo_improved_est: Dict[str, float] = {}
    simple_fs_mnpo_clf_est: Dict[str, float] = {}
    mnpo_fs_simple_clf_est: Dict[str, float] = {}
    mnpo_improved_cvar_est: Dict[str, float] = {}
    mnpo_experimental_fs_est: Dict[str, float] = {}
    mnpo_alt_screening_est: Dict[str, float] = {}
    for ds_id in validation_ids:
        base_t = float(val10_simple_hist.get(ds_id, 0.0))
        if base_t <= 0.0:
            spec = benchmark_datasets.get(ds_id)
            base_t = float(max(60.0, _dataset_weight(spec) * 220.0))
        simple_est[ds_id] = float(base_t)

        mnpo_t = float(val10_mnpo_hist.get(ds_id, 0.0))
        if mnpo_t <= 0.0:
            mnpo_t = float(max(base_t * 1.65, base_t + 90.0))
        mnpo_est[ds_id] = float(max(base_t * 1.20, mnpo_t))

        # Improved MNPO should be faster due to fewer oracles and no racing:
        # estimate ~60% of full MNPO runtime (2 oracles vs 5+, no racing, no rashomon)
        mnpo_improved_est[ds_id] = float(max(base_t * 1.10, mnpo_t * 0.60))

        # Profile C (simple FS + MNPO classifier): simple FS cost + slightly more
        # expensive classification (flaml + ensemble overhead). ~1.25× simple.
        simple_fs_mnpo_clf_est[ds_id] = float(base_t * 1.25)

        # Profile D (MNPO FS + simple classifier): MNPO FS cost + simple classifier.
        # ~same as mnpo_improved but classifier stage is cheaper.
        mnpo_fs_simple_clf_est[ds_id] = float(max(base_t * 1.10, mnpo_t * 0.55))

        # Profile E (MNPO improved + CVaR): slightly more than mnpo_improved
        # due to CVaR oracle overhead (~10% extra).
        mnpo_improved_cvar_est[ds_id] = float(max(base_t * 1.15, mnpo_t * 0.66))

        # Profile F (MNPO improved + experimental FS features): similar to B.
        # Racing saves some time, other FS features add modestly.
        mnpo_experimental_fs_est[ds_id] = float(max(base_t * 1.15, mnpo_t * 0.65))

        # Profile G (MNPO improved + alt screening/fitting): slightly more
        # expensive due to MPS estimator and DeepDRK knockoffs.
        mnpo_alt_screening_est[ds_id] = float(max(base_t * 1.20, mnpo_t * 0.70))

    # Partition datasets by combined expected runtime across all 7 profiles.
    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        combined = float(
            (
                simple_est[ds_id]
                + mnpo_improved_est[ds_id]
                + simple_fs_mnpo_clf_est[ds_id]
                + mnpo_fs_simple_clf_est[ds_id]
                + mnpo_improved_cvar_est[ds_id]
                + mnpo_experimental_fs_est[ds_id]
                + mnpo_alt_screening_est[ds_id]
            )
            * len(VALIDATION_SEEDS)
        )
        ds_items.append((str(ds_id), combined))
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_extra_args: Tuple[str, ...] = ("--emit-summary", "--compute-budget", "standard")

    # Shared flags (promoted Val-10 defaults — all three profiles use these)
    shared_stage_flags: Tuple[str, ...] = (
        "--df-family-set", "flex",
        "--df-compute-ad",
        "--df-compute-qq-pp",
        "--df-compute-dip",
        "--df-interval-likelihood",
        "--df-compute-crps",
        "--df-crps-uq-decomposition",
        "--df-lmoment-prescreen",
        "--df-lmoment-prescreen-max-candidates", "12",
        "--folding-method", "pls_da",
        "--enable-prefilter-rnaseq-nb-lrt",
        "--prefilter-rnaseq-nb-lrt-alpha", "0.10",
        "--enable-classifier-conformal",
        "--classifier-conformal-alpha", "0.10",
        "--classifier-conformal-calibration-fraction", "0.25",
        "--classifier-conformal-min-calibration", "20",
        "--enable-stage2-ratio-augmentation",
        "--stage2-ratio-max-features", "16",
        "--stage2-ratio-selection-method", "correlation",
        "--enable-model-cv-runtime-containment",
    )

    # Profile A: simple_control (identical to Val-10 simple_all_stages)
    simple_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--mnpo-performance-oracle-mode", "single",
        "--fs-oracle-weighting-mode", "uniform",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
        "--model-cv-runtime-max-candidates", "8",
    )

    # Profile B: mnpo_improved (MNPO with all post-Val-10 improvements)
    # Key changes vs Val-10 MNPO:
    #   - Banzhaf oracle weighting (not Shapley): lower weight variance
    #   - Only 2 oracles: performance + complexity (not 5+); eliminates tritrust noise
    #   - Selector penalty map active (default gates treeshap/dove/sparse_multinomial)
    #   - Train/test gap gating for tree models (threshold=0.15)
    #   - Tree complexity penalty enabled (strength=0.1)
    #   - No runtime racing, no rashomon, no QRE, no paradigm-aware prior
    #   - No UBayFS, no conformal UQ oracle, no interaction oracle
    #   - Simpler = fewer parameters to estimate from 5-fold CV
    mnpo_improved_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "mnpo_oracle",
        "--df-mnpo-include-crps",
        "--df-mnpo-include-preq",
        # No QRE, no oracle redundancy penalty, no tremble sensitivity
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        # No paradigm-aware prior
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        # Only perf + complexity oracles (T-R-248 perf_complexity preset)
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--fs-copula-derandomize-runs", "3",
        # No CVaR oracle — with only 2 base oracles, adding CVaR creates a 3-oracle system
        # that re-introduces the tritrust noise that caused Val-10 hard-tier losses
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        # Banzhaf oracle weighting (T-R-251) instead of Shapley
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        # No Bayesian shrinkage (remove over-regularization per HC1 findings)
        # No adaptive-sizing-variance-penalty
        # No UBayFS, no conformal UQ, no interaction oracle
        # No runtime racing (saves overhead; selector penalty map already gates weak methods)
        # No rashomon
        # Train/test gap gating (T-R-246)
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--classification-backend", "flaml",
        "--flaml-time-budget", "120",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "shrinkage_lda",
        "nsc", "pls_da_classifier", "nb", "vote_ensemble",
        "elastic_net_lr", "rf", "knn", "xgb", "lgbm", "extra_tree", "catboost",
        "--include-nsc-model",
        "--include-pls-da-model",
        "--include-lgbm-model",
        "--include-extra-tree-model",
        "--include-catboost-model",
        "--model-cv-runtime-max-candidates", "10",
    )

    # Profile C: simple_fs_mnpo_classifier (simple FS + MNPO ensemble classifier)
    # ★ KEY TEST: isolates classifier contribution.
    # If C ≈ B, MNPO's game-theoretic FS is unnecessary.
    simple_fs_mnpo_clf_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--mnpo-performance-oracle-mode", "single",
        "--fs-oracle-weighting-mode", "uniform",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        # MNPO ensemble classifier (same as Profile B)
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--classification-backend", "flaml",
        "--flaml-time-budget", "120",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "shrinkage_lda",
        "nsc", "pls_da_classifier", "nb", "vote_ensemble",
        "elastic_net_lr", "rf", "knn", "xgb", "lgbm", "extra_tree", "catboost",
        "--include-nsc-model",
        "--include-pls-da-model",
        "--include-lgbm-model",
        "--include-extra-tree-model",
        "--include-catboost-model",
        "--model-cv-runtime-max-candidates", "10",
        # Train/test gap gating for all profiles (T-R-246)
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
    )

    # Profile D: mnpo_fs_simple_classifier (MNPO FS + simple classifier)
    # Tests whether MNPO FS adds value when classifier is simple.
    # If D ≈ A, MNPO FS is overhead without value.
    mnpo_fs_simple_clf_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",  # Keep dist simple to isolate FS
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--fs-copula-derandomize-runs", "3",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
        # Simple single-model classifier (same as Profile A)
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
        "--model-cv-runtime-max-candidates", "8",
    )

    # Profile E: mnpo_improved_cvar (Profile B + CVaR tail-risk oracle)
    # Tests whether CVaR oracle improves worst-case BA.
    mnpo_improved_cvar_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "mnpo_oracle",
        "--df-mnpo-include-crps",
        "--df-mnpo-include-preq",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--fs-copula-derandomize-runs", "3",
        # CVaR oracle (T-R-250) — the key difference from Profile B
        "--fs-use-cvar-oracle",
        "--fs-cvar-alpha", "0.33",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--classification-backend", "flaml",
        "--flaml-time-budget", "120",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "shrinkage_lda",
        "nsc", "pls_da_classifier", "nb", "vote_ensemble",
        "elastic_net_lr", "rf", "knn", "xgb", "lgbm", "extra_tree", "catboost",
        "--include-nsc-model",
        "--include-pls-da-model",
        "--include-lgbm-model",
        "--include-extra-tree-model",
        "--include-catboost-model",
        "--model-cv-runtime-max-candidates", "10",
    )

    # Profile F: mnpo_improved_experimental_fs (Profile B + experimental FS features)
    # Tests 10 stacked experimental FS improvements vs vanilla B.
    # Contrast: F vs B reveals whether these FS refinements help.
    mnpo_experimental_fs_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "mnpo_oracle",
        "--df-mnpo-include-crps",
        "--df-mnpo-include-preq",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--fs-copula-derandomize-runs", "3",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--classification-backend", "flaml",
        "--flaml-time-budget", "120",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "shrinkage_lda",
        "nsc", "pls_da_classifier", "nb", "vote_ensemble",
        "elastic_net_lr", "rf", "knn", "xgb", "lgbm", "extra_tree", "catboost",
        "--include-nsc-model",
        "--include-pls-da-model",
        "--include-lgbm-model",
        "--include-extra-tree-model",
        "--include-catboost-model",
        "--model-cv-runtime-max-candidates", "10",
        # ---- Experimental FS features (stacked on B) ----
        "--enable-fs-wrapper-refine",
        "--enable-fs-stability-loss-guided-validation",
        "--enable-fs-adaptive-imbalance-score",
        "--enable-fs-runtime-racing",
        "--enable-fs-iterative-pruning-bounded-cpss-overlay",
        "--enable-fs-iterative-pruning-class-pareto-prefilter",
        "--enable-fs-iterative-pruning-class-pareto-stability-gate",
        "--enable-fs-ova-calibration",
        "--enable-fs-nsc-deep-shrinkage-search",
        "--fs-rank-aggregation-mode", "borda",
        "--enable-fs-copula-stabilizer-ebh",
    )

    # Profile G: mnpo_improved_alt_screening (Profile B + alternative screening/fitting)
    # Tests alternative screening (STIR), estimator (MPS), ratio features,
    # and knockoff generator (DeepDRK) vs vanilla B.
    # Contrast: G vs B reveals whether alt methods improve results.
    mnpo_alt_screening_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "mnpo_oracle",
        "--df-mnpo-include-crps",
        "--df-mnpo-include-preq",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        # Alt screening: STIR instead of evalue
        "--screening-enabled",
        "--screening-method", "stir",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--fs-copula-derandomize-runs", "3",
        # Alt knockoff generator: DeepDRK instead of default
        "--fs-copula-generator", "deepdrk",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--classification-backend", "flaml",
        "--flaml-time-budget", "120",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "shrinkage_lda",
        "nsc", "pls_da_classifier", "nb", "vote_ensemble",
        "elastic_net_lr", "rf", "knn", "xgb", "lgbm", "extra_tree", "catboost",
        "--include-nsc-model",
        "--include-pls-da-model",
        "--include-lgbm-model",
        "--include-extra-tree-model",
        "--include-catboost-model",
        "--model-cv-runtime-max-candidates", "10",
        # Alt estimator: MPS instead of MLE
        "--df-estimator", "mps",
        # Ratio features enabled
        "--enable-ratio-features",
    )

    common_job_params = dict(
        seeds=list(VALIDATION_SEEDS),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        progress_stall_watchdog_sec=1800,
        task_timeout_sec=21600,
        fs_method_timeout_sec=3600,
    )

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            profile_id="simple_control",
            fs_method_set="strict_plus_mrmr",
            extra_args=simple_flags,
            notes=(
                "Val-11 Profile A (control): simple dist + strict_plus_mrmr FS + "
                "sklearn legacy classifier. Identical to Val-10 simple_all_stages."
            ),
        ),
        BenchmarkProfile(
            profile_id="mnpo_improved",
            fs_method_set="mnpo_broad_all",
            extra_args=mnpo_improved_flags,
            notes=(
                "Val-11 Profile B: full improved MNPO — Banzhaf oracle weighting, "
                "2-oracle pruning (perf+complexity+diversity), selector penalty map, "
                "train/test gap gating, flaml mnpo_hybrid ensemble classifier."
            ),
        ),
        BenchmarkProfile(
            profile_id="simple_fs_mnpo_clf",
            fs_method_set="strict_plus_mrmr",
            extra_args=simple_fs_mnpo_clf_flags,
            notes=(
                "Val-11 Profile C (★ KEY TEST): simple FS (strict_plus_mrmr) + "
                "MNPO ensemble classifier (flaml mnpo_hybrid). Isolates classifier "
                "contribution. If C ≈ B, MNPO FS is unnecessary."
            ),
        ),
        BenchmarkProfile(
            profile_id="mnpo_fs_simple_clf",
            fs_method_set="mnpo_broad_all",
            extra_args=mnpo_fs_simple_clf_flags,
            notes=(
                "Val-11 Profile D: MNPO FS (mnpo_broad_all, banzhaf) + "
                "simple classifier (sklearn legacy). Isolates FS contribution. "
                "If D ≈ A, MNPO FS adds overhead without value."
            ),
        ),
        BenchmarkProfile(
            profile_id="mnpo_improved_cvar",
            fs_method_set="mnpo_broad_all",
            extra_args=mnpo_improved_cvar_flags,
            notes=(
                "Val-11 Profile E: same as B + CVaR tail-risk oracle (alpha=0.33). "
                "Tests whether CVaR improves worst-case BA."
            ),
        ),
        BenchmarkProfile(
            profile_id="mnpo_improved_experimental_fs",
            fs_method_set="mnpo_broad_all",
            extra_args=mnpo_experimental_fs_flags,
            notes=(
                "Val-11 Profile F: B + 11 experimental FS features (wrapper refine, "
                "stability-loss validation, adaptive imbalance score, runtime racing, "
                "CPSS overlay, class-Pareto prefilter+gate, OVA calibration, NSC deep "
                "shrinkage, Borda rank aggregation, copula stabilizer eBH). "
                "Contrast F vs B measures experimental FS feature value."
            ),
        ),
        BenchmarkProfile(
            profile_id="mnpo_improved_alt_screening",
            fs_method_set="mnpo_broad_all",
            extra_args=mnpo_alt_screening_flags,
            notes=(
                "Val-11 Profile G: B + alternative screening (STIR), estimator (MPS), "
                "ratio features, and knockoff generator (DeepDRK). "
                "Contrast G vs B measures alternative method value."
            ),
        ),
    ]

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError("Invalid validation11 profile(s):\n- " + "\n- ".join(invalid))

    runtime_by_profile = {
        "simple_control": simple_est,
        "mnpo_improved": mnpo_improved_est,
        "simple_fs_mnpo_clf": simple_fs_mnpo_clf_est,
        "mnpo_fs_simple_clf": mnpo_fs_simple_clf_est,
        "mnpo_improved_cvar": mnpo_improved_cvar_est,
        "mnpo_improved_experimental_fs": mnpo_experimental_fs_est,
        "mnpo_improved_alt_screening": mnpo_alt_screening_est,
    }

    jobs: List[Job] = []
    for prof in profiles:
        profile_runtime = dict(runtime_by_profile.get(prof.profile_id) or {})
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_seed_sec = 0.0
            for ds_id in ds_list:
                part_seed_sec += float(profile_runtime.get(ds_id, 180.0))
            part_weight = float(part_seed_sec * len(VALIDATION_SEEDS))
            jobs.append(
                _job(
                    f"val11/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    **common_job_params,
                )
            )
    return jobs


def _balanced_shard_assign_validation11_triples(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-11 by dataset partitions (group all 7 profiles per partition together)."""
    _VAL11_PROFILES = {
        "simple_control", "mnpo_improved", "simple_fs_mnpo_clf",
        "mnpo_fs_simple_clf", "mnpo_improved_cvar",
        "mnpo_improved_experimental_fs", "mnpo_improved_alt_screening",
    }
    profile_re = re.compile(
        r"^val11/(" + "|".join(re.escape(p) for p in _VAL11_PROFILES) + r")/(ds\d+)$"
    )
    grouped: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = profile_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        grouped.setdefault(part_id, {})[profile_id] = job

    bundle_items: List[Tuple[float, List[Job]]] = []
    for part_id, bundle in grouped.items():
        if len(bundle) < len(_VAL11_PROFILES):
            # Partial bundle still gets grouped together
            pass
        bundle_weight = sum(float(j.weight) for j in bundle.values())
        bundle_items.append((bundle_weight, list(bundle.values())))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for bundle_weight, bundle_jobs in sorted(bundle_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        for j in sorted(bundle_jobs, key=lambda j: j.job_id):
            shards[target].append(j.job_id)
        totals[target] += float(bundle_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


def build_jobs_validation12(
    *,
    dataset_shards: int = 6,
    val11_root: Optional[Path] = None,
    val10_root: Optional[Path] = None,
) -> List[Job]:
    """Validation-12: regime-conditional gating comparison.

    Profiles (7 total):
      - a_control: simple baseline (Profile A)
      - d_default: production default from Val-11 (Profile D)
      - d_gated: Profile D + regime gate/safeguards (T-R-257/258/259)
      - d_gated_cvar: d_gated + CVaR oracle (tail-risk hedge)
      - d_gated_oracle_slim: d_gated with reduced oracle complexity
        (single performance oracle, no eval proxy, no diversity oracle, + redundancy penalty)
      - d_gated_oracle_slim_fs030: d_gated_oracle_slim with fs_fraction=0.30
      - d_gated_oracle_slim_fs055: d_gated_oracle_slim with fs_fraction=0.55
    """
    VAL12_DATASETS: List[str] = [
        # Val-11 core (24)
        "arcene_nips03",
        "carcinom_11class",
        "cll_sub_111",
        "cns_pomeroy",
        "colon_alon",
        "cumida_brain_gse50161",
        "cumida_colorectal_gse44861",
        "cumida_leukemia_subtypes",
        "dlbcl_shipp",
        "gli_85",
        "glioma_50_4class",
        "hf_breast_ge_mubashir1837",
        "leukemia_1_72_3class",
        "leukemia_golub",
        "lymphoma_3",
        "madelon_nips03",
        "nci60_strict_holdout",
        "nci9_60_9class",
        "ovarian_petricoin",
        "prostate_singh",
        "srbct_khan",
        "tox_171",
        "xena_tcga_coad_cms",
        "xena_tcga_gbm",
        # Additional failure-regime coverage (runtime-bounded extension)
        "gcm_ramaswamy",
        "tumor11_su",
    ]

    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, _dataset_sets = _load_benchmark_registry()

    hf_meta = _load_hf_manifest_metadata()
    hf_ids = set(hf_meta.keys())
    if not hf_ids:
        raise RuntimeError("HF bundle manifest metadata is empty; cannot build validation12 plan.")

    validation_ids = [ds_id for ds_id in VAL12_DATASETS if ds_id in hf_ids]
    if not validation_ids:
        raise RuntimeError("No Val-12 datasets are available in HF bundle manifests.")
    missing_hf = [ds_id for ds_id in VAL12_DATASETS if ds_id not in hf_ids]
    if missing_hf:
        preview = ", ".join(missing_hf[:10])
        suffix = " ..." if len(missing_hf) > 10 else ""
        print(
            f"[validation12] Skipping {len(missing_hf)} dataset(s) not present in "
            f"HF manifests: {preview}{suffix}",
            file=sys.stderr,
        )

    for required in ("strict_plus_mrmr", "mnpo_broad_all"):
        if required not in fs_method_sets:
            raise RuntimeError(f"Missing required method set {required!r} in FS_METHOD_SETS.")

    if val11_root is None:
        val11_root = (
            REPO_ROOT
            / "run_artifacts"
            / "validation-11-v2"
            / "val11_4pods_live_20260227_183025"
        )
    val11_runtime = _load_runtime_hints_from_summaries(val11_root, phase_tag="val11")
    val11_a_hist = dict(val11_runtime.get("simple_control") or {})
    val11_d_hist = dict(val11_runtime.get("mnpo_fs_simple_clf") or {})

    if val10_root is None:
        val10_root = (
            REPO_ROOT
            / "run_artifacts"
            / "validation-10"
            / "val10_finished_done_only_pull_20260227_111010"
        )
    val10_runtime = _load_val10_runtime_hints_from_summaries(val10_root)
    val10_simple_hist = dict(val10_runtime.get("simple_all_stages") or {})
    val10_mnpo_hist = dict(val10_runtime.get("mnpo_all_stages") or {})

    a_est: Dict[str, float] = {}
    d_default_est: Dict[str, float] = {}
    d_gated_est: Dict[str, float] = {}
    d_gated_cvar_est: Dict[str, float] = {}
    d_gated_oracle_slim_est: Dict[str, float] = {}
    d_gated_oracle_slim_fs030_est: Dict[str, float] = {}
    d_gated_oracle_slim_fs055_est: Dict[str, float] = {}
    for ds_id in validation_ids:
        spec = benchmark_datasets.get(ds_id)
        ds_meta = dict(hf_meta.get(ds_id) or {})

        a_seed = float(val11_a_hist.get(ds_id, 0.0))
        if a_seed <= 0.0:
            a_seed = float(val10_simple_hist.get(ds_id, 0.0))
        if a_seed <= 0.0:
            a_seed = float(max(60.0, _dataset_weight(spec) * 220.0))
        a_est[ds_id] = float(a_seed)

        d_seed = float(val11_d_hist.get(ds_id, 0.0))
        if d_seed <= 0.0:
            v10_m = float(val10_mnpo_hist.get(ds_id, 0.0))
            if v10_m > 0.0:
                d_seed = float(max(a_seed * 1.10, v10_m * 0.55))
        if d_seed <= 0.0:
            d_seed = float(max(a_seed * 1.35, _dataset_weight(spec) * 260.0))
        d_default_est[ds_id] = float(d_seed)

        n_samples = ds_meta.get("n_samples")
        n_features = ds_meta.get("n_features")
        n_classes = ds_meta.get("n_classes")
        if n_classes is None:
            n_classes = _infer_dataset_class_count(ds_id, benchmark_datasets)
        tier = str(getattr(spec, "tier", "") or "").strip().lower()
        if not tier:
            tier = str(ds_meta.get("tier", "") or "").strip().lower()

        p_over_n = float(n_features / max(1, n_samples)) if isinstance(n_samples, int) and isinstance(n_features, int) else float("nan")
        samples_per_class = float(n_samples / max(1, int(n_classes))) if isinstance(n_samples, int) and isinstance(n_classes, int) and int(n_classes) > 0 else float("nan")
        very_hard_trigger = bool(tier == "very_hard")
        if np.isfinite(samples_per_class):
            very_hard_trigger = bool(very_hard_trigger or (samples_per_class < 15.0))
        low_p_over_n_trigger = bool(np.isfinite(p_over_n) and (p_over_n < 2.0))

        if low_p_over_n_trigger:
            # FS bypass should be materially faster than default D.
            d_gated_est[ds_id] = float(max(a_seed * 1.02, d_seed * 0.55))
        elif very_hard_trigger:
            # Very-hard fallback removes MNPO game optimization overhead.
            d_gated_est[ds_id] = float(max(a_seed * 1.08, d_seed * 0.75))
        else:
            # Non-gated datasets should be near parity with D_default.
            d_gated_est[ds_id] = float(max(a_seed * 1.05, d_seed * 0.95))

        if low_p_over_n_trigger:
            # CVaR overhead is small when low p/n bypass path dominates runtime.
            d_gated_cvar_est[ds_id] = float(max(a_seed * 1.03, d_gated_est[ds_id] * 1.02))
            d_gated_oracle_slim_est[ds_id] = float(max(a_seed * 1.01, d_gated_est[ds_id] * 0.93))
        elif very_hard_trigger:
            # CVaR can add moderate overhead on difficult folds; slim profile is faster.
            d_gated_cvar_est[ds_id] = float(max(a_seed * 1.12, d_gated_est[ds_id] * 1.08))
            d_gated_oracle_slim_est[ds_id] = float(max(a_seed * 1.05, d_gated_est[ds_id] * 0.88))
        else:
            # Non-gated tasks: CVaR slight overhead, slim profile near d_default runtime.
            d_gated_cvar_est[ds_id] = float(max(a_seed * 1.08, d_gated_est[ds_id] * 1.06))
            d_gated_oracle_slim_est[ds_id] = float(max(a_seed * 1.03, d_gated_est[ds_id] * 0.85))

        if low_p_over_n_trigger:
            # Low p/n path uses all-features fallback; fs_fraction has little/no impact.
            d_gated_oracle_slim_fs030_est[ds_id] = float(d_gated_oracle_slim_est[ds_id])
            d_gated_oracle_slim_fs055_est[ds_id] = float(d_gated_oracle_slim_est[ds_id])
        else:
            d_gated_oracle_slim_fs030_est[ds_id] = float(
                max(a_seed * 0.95, d_gated_oracle_slim_est[ds_id] * 0.88)
            )
            d_gated_oracle_slim_fs055_est[ds_id] = float(
                max(a_seed * 1.10, d_gated_oracle_slim_est[ds_id] * 1.12)
            )

    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        combined = float(
            (
                a_est[ds_id]
                + d_default_est[ds_id]
                + d_gated_est[ds_id]
                + d_gated_cvar_est[ds_id]
                + d_gated_oracle_slim_est[ds_id]
                + d_gated_oracle_slim_fs030_est[ds_id]
                + d_gated_oracle_slim_fs055_est[ds_id]
            )
            * len(VALIDATION_SEEDS)
        )
        ds_items.append((str(ds_id), combined))
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_extra_args: Tuple[str, ...] = ("--emit-summary", "--compute-budget", "standard")
    shared_stage_flags: Tuple[str, ...] = (
        "--df-family-set", "flex",
        "--df-compute-ad",
        "--df-compute-qq-pp",
        "--df-compute-dip",
        "--df-interval-likelihood",
        "--df-compute-crps",
        "--df-crps-uq-decomposition",
        "--df-lmoment-prescreen",
        "--df-lmoment-prescreen-max-candidates", "12",
        "--folding-method", "pls_da",
        "--enable-prefilter-rnaseq-nb-lrt",
        "--prefilter-rnaseq-nb-lrt-alpha", "0.10",
        "--enable-classifier-conformal",
        "--classifier-conformal-alpha", "0.10",
        "--classifier-conformal-calibration-fraction", "0.25",
        "--classifier-conformal-min-calibration", "20",
        "--enable-stage2-ratio-augmentation",
        "--stage2-ratio-max-features", "16",
        "--stage2-ratio-selection-method", "correlation",
        "--enable-model-cv-runtime-containment",
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
    )

    a_control_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--mnpo-performance-oracle-mode", "single",
        "--fs-oracle-weighting-mode", "uniform",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
        "--model-cv-runtime-max-candidates", "8",
    )

    d_default_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--fs-copula-derandomize-runs", "3",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
        "--model-cv-runtime-max-candidates", "8",
    )

    d_gated_flags: Tuple[str, ...] = (
        *d_default_flags,
        "--regime-gating-enabled",
        "--regime-gating-difficulty-source", "historical",
        "--regime-gating-target-tier", "very_hard",
        "--regime-gating-min-samples-per-class", "15",
        "--regime-gating-low-p-over-n-threshold", "2.0",
        "--regime-gating-simple-fs-method-set", "strict_plus_mrmr",
        "--regime-gating-very-hard-portfolio-max-methods", "4",
        "--regime-gating-very-hard-copula-derandomize-runs", "5",
        "--regime-gating-low-p-over-n-mode", "all_features",
    )

    d_gated_cvar_flags: Tuple[str, ...] = (
        *d_gated_flags,
        "--fs-use-cvar-oracle",
        "--fs-cvar-alpha", "0.33",
    )

    d_gated_oracle_slim_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--mnpo-performance-oracle-mode", "single",
        "--fs-copula-derandomize-runs", "3",
        "--fs-diversity-oracle-mode", "mi_redundancy",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        "--fs-use-oracle-redundancy-penalty",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
        "--model-cv-runtime-max-candidates", "8",
        "--regime-gating-enabled",
        "--regime-gating-difficulty-source", "historical",
        "--regime-gating-target-tier", "very_hard",
        "--regime-gating-min-samples-per-class", "15",
        "--regime-gating-low-p-over-n-threshold", "2.0",
        "--regime-gating-simple-fs-method-set", "strict_plus_mrmr",
        "--regime-gating-very-hard-portfolio-max-methods", "4",
        "--regime-gating-very-hard-copula-derandomize-runs", "5",
        "--regime-gating-low-p-over-n-mode", "all_features",
    )

    d_gated_oracle_slim_fs030_flags: Tuple[str, ...] = (
        *d_gated_oracle_slim_flags,
        "--fs-fraction", "0.30",
    )
    d_gated_oracle_slim_fs055_flags: Tuple[str, ...] = (
        *d_gated_oracle_slim_flags,
        "--fs-fraction", "0.55",
    )

    common_job_params = dict(
        seeds=list(VALIDATION_SEEDS),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        progress_stall_watchdog_sec=1800,
        task_timeout_sec=21600,
        fs_method_timeout_sec=3600,
    )

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            profile_id="a_control",
            fs_method_set="strict_plus_mrmr",
            extra_args=a_control_flags,
            notes="Val-12 A_control: simple baseline (Profile A).",
        ),
        BenchmarkProfile(
            profile_id="d_default",
            fs_method_set="mnpo_broad_all",
            extra_args=d_default_flags,
            notes="Val-12 D_default: production default (Profile D from Val-11).",
        ),
        BenchmarkProfile(
            profile_id="d_gated",
            fs_method_set="mnpo_broad_all",
            extra_args=d_gated_flags,
            notes="Val-12 D_gated: Profile D + regime-conditional safeguards.",
        ),
        BenchmarkProfile(
            profile_id="d_gated_cvar",
            fs_method_set="mnpo_broad_all",
            extra_args=d_gated_cvar_flags,
            notes="Val-12 D_gated_cvar: D_gated + CVaR oracle (tail-risk hedge).",
        ),
        BenchmarkProfile(
            profile_id="d_gated_oracle_slim",
            fs_method_set="mnpo_broad_all",
            extra_args=d_gated_oracle_slim_flags,
            notes=(
                "Val-12 D_gated_oracle_slim: D_gated with reduced oracle complexity "
                "(single performance oracle, no eval-proxy/diversity, redundancy penalty)."
            ),
        ),
        BenchmarkProfile(
            profile_id="d_gated_oracle_slim_fs030",
            fs_method_set="mnpo_broad_all",
            extra_args=d_gated_oracle_slim_fs030_flags,
            notes="Val-12 D_gated_oracle_slim with fs_fraction=0.30.",
        ),
        BenchmarkProfile(
            profile_id="d_gated_oracle_slim_fs055",
            fs_method_set="mnpo_broad_all",
            extra_args=d_gated_oracle_slim_fs055_flags,
            notes="Val-12 D_gated_oracle_slim with fs_fraction=0.55.",
        ),
    ]

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError("Invalid validation12 profile(s):\n- " + "\n- ".join(invalid))

    runtime_by_profile = {
        "a_control": a_est,
        "d_default": d_default_est,
        "d_gated": d_gated_est,
        "d_gated_cvar": d_gated_cvar_est,
        "d_gated_oracle_slim": d_gated_oracle_slim_est,
        "d_gated_oracle_slim_fs030": d_gated_oracle_slim_fs030_est,
        "d_gated_oracle_slim_fs055": d_gated_oracle_slim_fs055_est,
    }

    jobs: List[Job] = []
    for prof in profiles:
        profile_runtime = dict(runtime_by_profile.get(prof.profile_id) or {})
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_seed_sec = 0.0
            for ds_id in ds_list:
                part_seed_sec += float(profile_runtime.get(ds_id, 180.0))
            part_weight = float(part_seed_sec * len(VALIDATION_SEEDS))
            jobs.append(
                _job(
                    f"val12/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    **common_job_params,
                )
            )
    return jobs


def _balanced_shard_assign_validation12_triples(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-12 by dataset partitions (group all profile bundles per partition)."""
    _VAL12_PROFILES = {
        "a_control",
        "d_default",
        "d_gated",
        "d_gated_cvar",
        "d_gated_oracle_slim",
        "d_gated_oracle_slim_fs030",
        "d_gated_oracle_slim_fs055",
    }
    profile_re = re.compile(r"^val12/(" + "|".join(re.escape(p) for p in _VAL12_PROFILES) + r")/(ds\d+)$")
    grouped: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = profile_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        grouped.setdefault(part_id, {})[profile_id] = job

    bundle_items: List[Tuple[float, List[Job]]] = []
    for part_id, bundle in grouped.items():
        bundle_weight = sum(float(j.weight) for j in bundle.values())
        bundle_items.append((bundle_weight, list(bundle.values())))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for bundle_weight, bundle_jobs in sorted(bundle_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        for j in sorted(bundle_jobs, key=lambda j: j.job_id):
            shards[target].append(j.job_id)
        totals[target] += float(bundle_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


def build_jobs_validation13(
    *,
    dataset_shards: int = 8,
    val12_root: Optional[Path] = None,
    val11_root: Optional[Path] = None,
) -> List[Job]:
    """Validation-13: gate fixes + profile expansion + ablation + FS-fraction sweep.

    Profiles (13 total):
      Core:
      - a_control: Val-12 A baseline
      - d_default: Val-12 D baseline (production reference)
      - d_v13: primary Val-13 profile (all improvements: gate fixes + BH + stability + copula5 + variance floor)
      - g_alt_screening: Val-11-style alternative screening (STIR + MPS + ratio + DeepDRK)
      Ablation:
      - d_v13_gates_only: D_default + gate fixes only (no BH, no stability, no variance floor, copula=3)
      - d_v13_no_stability: D_v13 minus stability-weighted aggregation
      - d_v13_no_bh: D_v13 minus BH prefilter
      - d_v13_pareto: D_v13 + Pareto portfolio sizing
      - d_v13_multiclass: D_v13 with lower extreme-multiclass threshold (c≥6)
      FS sweep:
      - d_v13_fs025/fs040/fs055: 3-fraction FS sweep on d_v13
      Regime probe:
      - f_experimental_fs: Profile F from Val-11 (experimental FS features)
    """
    VAL13_CORE_DATASETS: List[str] = [
        # Val-12 core (30)
        "arcene_nips03",
        "carcinom_11class",
        "cll_sub_111",
        "cns_pomeroy",
        "colon_alon",
        "cumida_brain_gse50161",
        "cumida_colorectal_gse44861",
        "cumida_leukemia_subtypes",
        "dlbcl_shipp",
        "dorothea_nips03",
        "gcm_ramaswamy",
        "gli_85",
        "glioma_50_4class",
        "hf_breast_ge_mubashir1837",
        "leukemia_1_72_3class",
        "leukemia_golub",
        "lymphoma_3",
        "lymphoma_9",
        "madelon_nips03",
        "nci60_strict_holdout",
        "nci9_60_9class",
        "ovarian_petricoin",
        "prostate_singh",
        "srbct_khan",
        "tox_171",
        "tumor11_su",
        "xena_tcga_coad_cms",
        "xena_tcga_gbm",
        "xena_tcga_lgg",
        "xena_tcga_skcm",
    ]
    VAL13_EXTENSION_DATASETS: List[str] = [
        # Gap-filling extension (5)
        "nci_61_8class",
        "tumor9_openml",
        "cumida_gastric_gse54129",
        "cumida_renal_gse53757",
        "breast_vantveer",
    ]
    VAL13_DATASETS = VAL13_CORE_DATASETS + VAL13_EXTENSION_DATASETS

    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, _dataset_sets = _load_benchmark_registry()

    hf_meta = _load_hf_manifest_metadata()
    hf_ids = set(hf_meta.keys())
    if not hf_ids:
        raise RuntimeError("HF bundle manifest metadata is empty; cannot build validation13 plan.")

    validation_ids = [ds_id for ds_id in VAL13_DATASETS if ds_id in hf_ids]
    if not validation_ids:
        raise RuntimeError("No Val-13 datasets are available in HF bundle manifests.")
    missing_hf = [ds_id for ds_id in VAL13_DATASETS if ds_id not in hf_ids]
    if missing_hf:
        preview = ", ".join(missing_hf[:10])
        suffix = " ..." if len(missing_hf) > 10 else ""
        print(
            f"[validation13] Skipping {len(missing_hf)} dataset(s) not present in "
            f"HF manifests: {preview}{suffix}",
            file=sys.stderr,
        )

    for required in ("strict_plus_mrmr", "mnpo_broad_all"):
        if required not in fs_method_sets:
            raise RuntimeError(f"Missing required method set {required!r} in FS_METHOD_SETS.")

    if val12_root is None:
        val12_root = (
            REPO_ROOT
            / "run_artifacts"
            / "validation-12"
            / "val12_4pods_live_20260228_013532"
        )
    val12_runtime = _load_runtime_hints_from_summaries(val12_root, phase_tag="val12")
    val12_a_hist = dict(val12_runtime.get("a_control") or {})
    val12_d_hist = dict(val12_runtime.get("d_default") or {})
    val12_gated_hist = dict(val12_runtime.get("d_gated") or {})

    if val11_root is None:
        val11_root = (
            REPO_ROOT
            / "run_artifacts"
            / "validation-11-v2"
            / "val11_4pods_live_20260227_183025"
        )
    val11_runtime = _load_runtime_hints_from_summaries(val11_root, phase_tag="val11")
    val11_g_hist = dict(val11_runtime.get("mnpo_improved_alt_screening") or {})

    a_est: Dict[str, float] = {}
    d_default_est: Dict[str, float] = {}
    d_v13_est: Dict[str, float] = {}
    d_v13_gates_only_est: Dict[str, float] = {}
    d_v13_no_stability_est: Dict[str, float] = {}
    d_v13_no_bh_est: Dict[str, float] = {}
    d_v13_pareto_est: Dict[str, float] = {}
    d_v13_multiclass_est: Dict[str, float] = {}
    g_alt_screening_est: Dict[str, float] = {}
    f_experimental_fs_est: Dict[str, float] = {}
    d_v13_fs025_est: Dict[str, float] = {}
    d_v13_fs040_est: Dict[str, float] = {}
    d_v13_fs055_est: Dict[str, float] = {}

    for ds_id in validation_ids:
        spec = benchmark_datasets.get(ds_id)
        ds_meta = dict(hf_meta.get(ds_id) or {})

        base_t = float(val12_a_hist.get(ds_id, 0.0))
        if base_t <= 0.0:
            base_t = float(val12_d_hist.get(ds_id, 0.0) * 0.65)
        if base_t <= 0.0:
            base_t = float(max(60.0, _dataset_weight(spec) * 220.0))
        a_est[ds_id] = float(base_t)

        d_seed = float(val12_d_hist.get(ds_id, 0.0))
        if d_seed <= 0.0:
            d_seed = float(val12_gated_hist.get(ds_id, 0.0))
        if d_seed <= 0.0:
            d_seed = float(max(base_t * 1.35, _dataset_weight(spec) * 260.0))
        d_default_est[ds_id] = float(d_seed)

        n_samples = ds_meta.get("n_samples")
        n_features = ds_meta.get("n_features")
        p_over_n = float(n_features / max(1, n_samples)) if isinstance(n_samples, int) and isinstance(n_features, int) else float("nan")
        low_p_over_n_trigger = bool(np.isfinite(p_over_n) and (p_over_n < 2.0))

        # D_v13 is expected to be slightly slower than D_default due to extra safeguards.
        d_v13_est[ds_id] = float(max(base_t * 1.15, d_seed * 1.08))
        d_v13_gates_only_est[ds_id] = float(max(base_t * 1.02, d_seed * 1.02))
        d_v13_no_stability_est[ds_id] = float(max(base_t * 1.12, d_seed * 1.05))
        d_v13_no_bh_est[ds_id] = float(max(base_t * 1.12, d_seed * 1.05))
        d_v13_pareto_est[ds_id] = float(max(base_t * 1.20, d_v13_est[ds_id] * 1.10))
        d_v13_multiclass_est[ds_id] = float(max(base_t * 1.16, d_v13_est[ds_id] * 1.02))

        g_hist = float(val11_g_hist.get(ds_id, 0.0))
        if g_hist <= 0.0:
            g_hist = float(max(base_t * 1.25, d_seed * 0.95))
        g_alt_screening_est[ds_id] = float(g_hist)

        # F_experimental_fs: comparable to G (MNPO classifier + experimental FS features)
        f_experimental_fs_est[ds_id] = float(max(base_t * 1.15, g_hist * 0.95))

        # FS-fraction sweep (reduced from 5 to 3: 0.25, 0.40, 0.55).
        if low_p_over_n_trigger:
            # Low p/n gate uses fast univariate filter; fs_fraction impact is expected to be weak.
            d_v13_fs025_est[ds_id] = float(d_v13_est[ds_id])
            d_v13_fs040_est[ds_id] = float(d_v13_est[ds_id])
            d_v13_fs055_est[ds_id] = float(d_v13_est[ds_id])
        else:
            d_v13_fs025_est[ds_id] = float(max(base_t * 0.95, d_v13_est[ds_id] * 0.86))
            d_v13_fs040_est[ds_id] = float(d_v13_est[ds_id])
            d_v13_fs055_est[ds_id] = float(max(base_t * 1.12, d_v13_est[ds_id] * 1.16))

    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        combined = float(
            (
                a_est[ds_id]
                + d_default_est[ds_id]
                + d_v13_est[ds_id]
                + d_v13_gates_only_est[ds_id]
                + d_v13_no_stability_est[ds_id]
                + d_v13_no_bh_est[ds_id]
                + d_v13_pareto_est[ds_id]
                + d_v13_multiclass_est[ds_id]
                + g_alt_screening_est[ds_id]
                + f_experimental_fs_est[ds_id]
                + d_v13_fs025_est[ds_id]
                + d_v13_fs040_est[ds_id]
                + d_v13_fs055_est[ds_id]
            )
            * len(VALIDATION_SEEDS)
        )
        ds_items.append((str(ds_id), combined))
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_extra_args: Tuple[str, ...] = ("--emit-summary", "--compute-budget", "standard")
    shared_stage_flags: Tuple[str, ...] = (
        "--df-family-set", "flex",
        "--df-compute-ad",
        "--df-compute-qq-pp",
        "--df-compute-dip",
        "--df-interval-likelihood",
        "--df-compute-crps",
        "--df-crps-uq-decomposition",
        "--df-lmoment-prescreen",
        "--df-lmoment-prescreen-max-candidates", "12",
        "--folding-method", "pls_da",
        "--enable-prefilter-rnaseq-nb-lrt",
        "--prefilter-rnaseq-nb-lrt-alpha", "0.10",
        "--enable-classifier-conformal",
        "--classifier-conformal-alpha", "0.10",
        "--classifier-conformal-calibration-fraction", "0.25",
        "--classifier-conformal-min-calibration", "20",
        "--enable-stage2-ratio-augmentation",
        "--stage2-ratio-max-features", "16",
        "--stage2-ratio-selection-method", "correlation",
        "--enable-model-cv-runtime-containment",
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
    )

    a_control_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--mnpo-performance-oracle-mode", "single",
        "--fs-oracle-weighting-mode", "uniform",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
        "--model-cv-runtime-max-candidates", "8",
    )

    d_default_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--fs-copula-derandomize-runs", "3",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
        "--model-cv-runtime-max-candidates", "8",
    )

    d_v13_flags: Tuple[str, ...] = (
        *d_default_flags,
        "--regime-gating-enabled",
        "--regime-gating-difficulty-source", "historical",
        "--regime-gating-target-tier", "very_hard",
        "--regime-gating-min-samples-per-class", "7",
        "--regime-gating-very-hard-min-classes", "5",
        "--regime-gating-low-p-over-n-threshold", "2.0",
        "--regime-gating-simple-fs-method-set", "strict_plus_mrmr",
        "--regime-gating-very-hard-portfolio-max-methods", "4",
        "--regime-gating-very-hard-copula-derandomize-runs", "5",
        "--regime-gating-low-p-over-n-mode", "fast_univariate_filter",
        "--regime-gating-extreme-multiclass-threshold", "8",
        "--prefilter-bh-ttest-enabled",
        "--prefilter-variance-floor-enabled",
        "--fs-copula-derandomize-runs", "5",
        "--fs-stability-weighted-aggregation-enabled",
    )

    d_v13_pareto_flags: Tuple[str, ...] = (
        *d_v13_flags,
        "--fs-pareto-portfolio-sizing-enabled",
    )
    d_v13_multiclass_flags: Tuple[str, ...] = (
        *d_v13_flags,
        "--regime-gating-extreme-multiclass-threshold", "6",
    )

    g_alt_screening_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "mnpo_oracle",
        "--df-mnpo-include-crps",
        "--df-mnpo-include-preq",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "stir",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--fs-copula-derandomize-runs", "3",
        "--fs-copula-generator", "deepdrk",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--classification-backend", "flaml",
        "--flaml-time-budget", "120",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "shrinkage_lda",
        "nsc", "pls_da_classifier", "nb", "vote_ensemble",
        "elastic_net_lr", "rf", "knn", "xgb", "lgbm", "extra_tree", "catboost",
        "--include-nsc-model",
        "--include-pls-da-model",
        "--include-lgbm-model",
        "--include-extra-tree-model",
        "--include-catboost-model",
        "--model-cv-runtime-max-candidates", "10",
        "--df-estimator", "mps",
        "--enable-ratio-features",
    )

    # --- Ablation profiles (NEW) ---

    # D_v13_gates_only: D_default + only gate fixes (no BH, no stability, no variance floor, copula=3)
    d_v13_gates_only_flags: Tuple[str, ...] = (
        *d_default_flags,
        "--regime-gating-enabled",
        "--regime-gating-difficulty-source", "historical",
        "--regime-gating-target-tier", "very_hard",
        "--regime-gating-min-samples-per-class", "7",
        "--regime-gating-very-hard-min-classes", "5",
        "--regime-gating-low-p-over-n-threshold", "2.0",
        "--regime-gating-simple-fs-method-set", "strict_plus_mrmr",
        "--regime-gating-very-hard-portfolio-max-methods", "4",
        "--regime-gating-very-hard-copula-derandomize-runs", "5",
        "--regime-gating-low-p-over-n-mode", "fast_univariate_filter",
        "--regime-gating-extreme-multiclass-threshold", "8",
        "--no-prefilter-bh-ttest",
        "--no-prefilter-variance-floor",
    )

    # D_v13_no_stability: D_v13 minus stability-weighted aggregation
    d_v13_no_stability_flags: Tuple[str, ...] = (
        *d_default_flags,
        "--regime-gating-enabled",
        "--regime-gating-difficulty-source", "historical",
        "--regime-gating-target-tier", "very_hard",
        "--regime-gating-min-samples-per-class", "7",
        "--regime-gating-very-hard-min-classes", "5",
        "--regime-gating-low-p-over-n-threshold", "2.0",
        "--regime-gating-simple-fs-method-set", "strict_plus_mrmr",
        "--regime-gating-very-hard-portfolio-max-methods", "4",
        "--regime-gating-very-hard-copula-derandomize-runs", "5",
        "--regime-gating-low-p-over-n-mode", "fast_univariate_filter",
        "--regime-gating-extreme-multiclass-threshold", "8",
        "--prefilter-bh-ttest-enabled",
        "--prefilter-variance-floor-enabled",
        "--fs-copula-derandomize-runs", "5",
        # NOTE: fs-stability-weighted-aggregation-enabled intentionally OMITTED
    )

    # D_v13_no_bh: D_v13 minus BH prefilter (variance floor still on)
    d_v13_no_bh_flags: Tuple[str, ...] = (
        *d_default_flags,
        "--regime-gating-enabled",
        "--regime-gating-difficulty-source", "historical",
        "--regime-gating-target-tier", "very_hard",
        "--regime-gating-min-samples-per-class", "7",
        "--regime-gating-very-hard-min-classes", "5",
        "--regime-gating-low-p-over-n-threshold", "2.0",
        "--regime-gating-simple-fs-method-set", "strict_plus_mrmr",
        "--regime-gating-very-hard-portfolio-max-methods", "4",
        "--regime-gating-very-hard-copula-derandomize-runs", "5",
        "--regime-gating-low-p-over-n-mode", "fast_univariate_filter",
        "--regime-gating-extreme-multiclass-threshold", "8",
        "--no-prefilter-bh-ttest",
        "--prefilter-variance-floor-enabled",
        "--fs-copula-derandomize-runs", "5",
        "--fs-stability-weighted-aggregation-enabled",
    )

    # F_experimental_fs: Profile F from Val-11 (experimental FS features)
    f_experimental_fs_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "mnpo_oracle",
        "--df-mnpo-include-crps",
        "--df-mnpo-include-preq",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--fs-copula-derandomize-runs", "3",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--classification-backend", "flaml",
        "--flaml-time-budget", "120",
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "shrinkage_lda",
        "nsc", "pls_da_classifier", "nb", "vote_ensemble",
        "elastic_net_lr", "rf", "knn", "xgb", "lgbm", "extra_tree", "catboost",
        "--include-nsc-model",
        "--include-pls-da-model",
        "--include-lgbm-model",
        "--include-extra-tree-model",
        "--include-catboost-model",
        "--model-cv-runtime-max-candidates", "10",
        # ---- Experimental FS features (stacked on B) ----
        "--enable-fs-wrapper-refine",
        "--enable-fs-stability-loss-guided-validation",
        "--enable-fs-adaptive-imbalance-score",
        "--enable-fs-runtime-racing",
        "--enable-fs-iterative-pruning-bounded-cpss-overlay",
        "--enable-fs-iterative-pruning-class-pareto-prefilter",
        "--enable-fs-iterative-pruning-class-pareto-stability-gate",
        "--enable-fs-ova-calibration",
        "--enable-fs-nsc-deep-shrinkage-search",
        "--fs-rank-aggregation-mode", "borda",
        "--enable-fs-copula-stabilizer-ebh",
    )

    # --- FS-fraction sweep (reduced from 5 to 3) ---
    d_v13_fs025_flags: Tuple[str, ...] = (*d_v13_flags, "--fs-fraction", "0.25")
    d_v13_fs040_flags: Tuple[str, ...] = (*d_v13_flags, "--fs-fraction", "0.40")
    d_v13_fs055_flags: Tuple[str, ...] = (*d_v13_flags, "--fs-fraction", "0.55")

    common_job_params = dict(
        seeds=list(VALIDATION_SEEDS),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        progress_stall_watchdog_sec=1800,
        task_timeout_sec=21600,
        fs_method_timeout_sec=3600,
    )

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            profile_id="a_control",
            fs_method_set="strict_plus_mrmr",
            extra_args=a_control_flags,
            notes="Val-13 A_control baseline.",
        ),
        BenchmarkProfile(
            profile_id="d_default",
            fs_method_set="mnpo_broad_all",
            extra_args=d_default_flags,
            notes="Val-13 production reference (Profile D default).",
        ),
        BenchmarkProfile(
            profile_id="d_v13",
            fs_method_set="mnpo_broad_all",
            extra_args=d_v13_flags,
            notes="Val-13 primary profile with gate fixes + BH + stability weighting.",
        ),
        BenchmarkProfile(
            profile_id="g_alt_screening",
            fs_method_set="mnpo_broad_all",
            extra_args=g_alt_screening_flags,
            notes="Val-13 G revival (STIR + MPS + ratio features + DeepDRK).",
        ),
        # --- Ablation profiles ---
        BenchmarkProfile(
            profile_id="d_v13_gates_only",
            fs_method_set="mnpo_broad_all",
            extra_args=d_v13_gates_only_flags,
            notes="Val-13 ablation: D_default + gate fixes only (no BH/stability/variance-floor).",
        ),
        BenchmarkProfile(
            profile_id="d_v13_no_stability",
            fs_method_set="mnpo_broad_all",
            extra_args=d_v13_no_stability_flags,
            notes="Val-13 ablation: D_v13 minus stability-weighted aggregation.",
        ),
        BenchmarkProfile(
            profile_id="d_v13_no_bh",
            fs_method_set="mnpo_broad_all",
            extra_args=d_v13_no_bh_flags,
            notes="Val-13 ablation: D_v13 minus BH prefilter.",
        ),
        BenchmarkProfile(
            profile_id="d_v13_pareto",
            fs_method_set="mnpo_broad_all",
            extra_args=d_v13_pareto_flags,
            notes="Val-13 D_v13 + Pareto portfolio sizing.",
        ),
        BenchmarkProfile(
            profile_id="d_v13_multiclass",
            fs_method_set="mnpo_broad_all",
            extra_args=d_v13_multiclass_flags,
            notes="Val-13 D_v13 multiclass sensitivity (threshold=6).",
        ),
        # --- FS-fraction sweep (3 fractions) ---
        BenchmarkProfile(
            profile_id="d_v13_fs025",
            fs_method_set="mnpo_broad_all",
            extra_args=d_v13_fs025_flags,
            notes="Val-13 D_v13 ratio sweep fs_fraction=0.25.",
        ),
        BenchmarkProfile(
            profile_id="d_v13_fs040",
            fs_method_set="mnpo_broad_all",
            extra_args=d_v13_fs040_flags,
            notes="Val-13 D_v13 ratio sweep fs_fraction=0.40.",
        ),
        BenchmarkProfile(
            profile_id="d_v13_fs055",
            fs_method_set="mnpo_broad_all",
            extra_args=d_v13_fs055_flags,
            notes="Val-13 D_v13 ratio sweep fs_fraction=0.55.",
        ),
        # --- Regime probe ---
        BenchmarkProfile(
            profile_id="f_experimental_fs",
            fs_method_set="mnpo_broad_all",
            extra_args=f_experimental_fs_flags,
            notes="Val-13 Profile F regime probe (experimental FS features from Val-11).",
        ),
    ]

    _assert_no_implicit_true_omissions(
        profiles=profiles,
        required_negations_by_profile={
            "d_v13_no_bh": ("--no-prefilter-bh-ttest",),
            "d_v13_gates_only": ("--no-prefilter-bh-ttest", "--no-prefilter-variance-floor"),
        },
        required_value_overrides_by_profile={
            "d_v13": (("--fs-copula-derandomize-runs", "5"),),
            "d_v13_no_bh": (("--fs-copula-derandomize-runs", "5"),),
            "d_v13_no_stability": (("--fs-copula-derandomize-runs", "5"),),
            "d_v13_pareto": (("--fs-copula-derandomize-runs", "5"),),
            "d_v13_multiclass": (("--fs-copula-derandomize-runs", "5"),),
            "d_v13_fs025": (("--fs-copula-derandomize-runs", "5"),),
            "d_v13_fs040": (("--fs-copula-derandomize-runs", "5"),),
            "d_v13_fs055": (("--fs-copula-derandomize-runs", "5"),),
            "d_v13_gates_only": (("--fs-copula-derandomize-runs", "3"),),
        },
        context="validation13",
    )

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError("Invalid validation13 profile(s):\n- " + "\n- ".join(invalid))

    runtime_by_profile = {
        "a_control": a_est,
        "d_default": d_default_est,
        "d_v13": d_v13_est,
        "d_v13_gates_only": d_v13_gates_only_est,
        "d_v13_no_stability": d_v13_no_stability_est,
        "d_v13_no_bh": d_v13_no_bh_est,
        "d_v13_pareto": d_v13_pareto_est,
        "d_v13_multiclass": d_v13_multiclass_est,
        "g_alt_screening": g_alt_screening_est,
        "f_experimental_fs": f_experimental_fs_est,
        "d_v13_fs025": d_v13_fs025_est,
        "d_v13_fs040": d_v13_fs040_est,
        "d_v13_fs055": d_v13_fs055_est,
    }

    jobs: List[Job] = []
    for prof in profiles:
        profile_runtime = dict(runtime_by_profile.get(prof.profile_id) or {})
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_seed_sec = 0.0
            for ds_id in ds_list:
                part_seed_sec += float(profile_runtime.get(ds_id, 180.0))
            part_weight = float(part_seed_sec * len(VALIDATION_SEEDS))
            jobs.append(
                _job(
                    f"val13/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    **common_job_params,
                )
            )
    return jobs


def _balanced_shard_assign_validation13_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-13 by dataset partitions (group all profile bundles per partition)."""
    _VAL13_PROFILES = {
        "a_control",
        "d_default",
        "d_v13",
        "d_v13_gates_only",
        "d_v13_no_stability",
        "d_v13_no_bh",
        "d_v13_pareto",
        "d_v13_multiclass",
        "g_alt_screening",
        "f_experimental_fs",
        "d_v13_fs025",
        "d_v13_fs040",
        "d_v13_fs055",
    }
    profile_re = re.compile(r"^val13/(" + "|".join(re.escape(p) for p in _VAL13_PROFILES) + r")/(ds\d+)$")
    grouped: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = profile_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        grouped.setdefault(part_id, {})[profile_id] = job

    bundle_items: List[Tuple[float, List[Job]]] = []
    for part_id, bundle in grouped.items():
        bundle_weight = sum(float(j.weight) for j in bundle.values())
        bundle_items.append((bundle_weight, list(bundle.values())))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for bundle_weight, bundle_jobs in sorted(bundle_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        for j in sorted(bundle_jobs, key=lambda j: j.job_id):
            shards[target].append(j.job_id)
        totals[target] += float(bundle_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


# ---------------------------------------------------------------------------
# Validation-14 feature-effect matrix + activation smoke
# ---------------------------------------------------------------------------

VALIDATION14_DATASETS: List[str] = [
    # Same 35 datasets as Val-13.
    "arcene_nips03",
    "carcinom_11class",
    "cll_sub_111",
    "cns_pomeroy",
    "colon_alon",
    "cumida_brain_gse50161",
    "cumida_colorectal_gse44861",
    "cumida_leukemia_subtypes",
    "dlbcl_shipp",
    "dorothea_nips03",
    "gcm_ramaswamy",
    "gli_85",
    "glioma_50_4class",
    "hf_breast_ge_mubashir1837",
    "leukemia_1_72_3class",
    "leukemia_golub",
    "lymphoma_3",
    "lymphoma_9",
    "madelon_nips03",
    "nci60_strict_holdout",
    "nci9_60_9class",
    "ovarian_petricoin",
    "prostate_singh",
    "srbct_khan",
    "tox_171",
    "tumor11_su",
    "xena_tcga_coad_cms",
    "xena_tcga_gbm",
    "xena_tcga_lgg",
    "xena_tcga_skcm",
    "nci_61_8class",
    "tumor9_openml",
    "cumida_gastric_gse54129",
    "cumida_renal_gse53757",
    "breast_vantveer",
]

VALIDATION14_ACTIVATION_DATASETS: List[str] = [
    # Focused subset for fast activation checks.
    "dorothea_nips03",
    "madelon_nips03",
    "gcm_ramaswamy",
    "nci9_60_9class",
    "xena_tcga_lgg",
    "leukemia_golub",
]

VALIDATION14_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "a_control": {"anchor": "yes", "contrast_ref": "v14_ref", "effect": "legacy_simple_anchor"},
    "d_default": {"anchor": "yes", "contrast_ref": "v14_ref", "effect": "production_anchor"},
    "v14_ref": {"anchor": "yes", "contrast_ref": "d_default", "effect": "corrected_reference"},
    "v14_no_bh": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "bh_prefilter"},
    "v14_no_varfloor": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "variance_floor"},
    "v14_no_copula5": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "copula_5_vs_3"},
    "v14_add_stability": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "stability_weighting"},
    "v14_add_pareto": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "pareto_sizing"},
    "v14_no_extreme_multiclass": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "extreme_multiclass_gate"},
    "v14_tritrust": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "oracle_weighting_mode"},
    "v14_tabpfn": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "tabpfn_candidate"},
    "v14_ipss": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "ipss_method"},
    "v14_ipss_eats": {"anchor": "no", "contrast_ref": "v14_ipss", "effect": "eats_thresholding"},
    "v14_group_sparse": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "group_sparse_method"},
    "v14_batch_combat": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "combat_batch_correction"},
    "v14_mapie_aps": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "mapie_aps"},
    "v14_mapie_raps": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "mapie_raps"},
    "v14_mapie_cross": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "mapie_cross"},
    "v14_multiomics_adapter": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "multiomics_adapter"},
    "v14_gates_only": {"anchor": "no", "contrast_ref": "v14_ref", "effect": "gates_only_bundle"},
}


def _build_jobs_validation14_common(
    *,
    dataset_ids: Sequence[str],
    dataset_shards: int,
    val13_root: Optional[Path],
    seeds: Sequence[int],
    phase_prefix: str,
    task_timeout_sec: int,
    fs_method_timeout_sec: int,
) -> List[Job]:
    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, _dataset_sets = _load_benchmark_registry()
    hf_meta = _load_hf_manifest_metadata()
    hf_ids = set(hf_meta.keys())
    if not hf_ids:
        raise RuntimeError("HF bundle manifest metadata is empty; cannot build validation14 plan.")

    validation_ids = [str(ds) for ds in dataset_ids if str(ds) in hf_ids]
    if not validation_ids:
        raise RuntimeError(f"No datasets are available in HF bundle manifests for {phase_prefix}.")

    missing_hf = [str(ds) for ds in dataset_ids if str(ds) not in hf_ids]
    if missing_hf:
        preview = ", ".join(missing_hf[:10])
        suffix = " ..." if len(missing_hf) > 10 else ""
        print(
            f"[{phase_prefix}] Skipping {len(missing_hf)} dataset(s) not present in "
            f"HF manifests: {preview}{suffix}",
            file=sys.stderr,
        )

    required_method_sets = (
        "strict_plus_mrmr",
        "mnpo_broad_all",
        "mnpo_v14_core",
        "mnpo_v14_core_plus_ipss",
        "mnpo_v14_core_plus_group_sparse",
    )
    for required in required_method_sets:
        if required not in fs_method_sets:
            raise RuntimeError(f"Missing required method set {required!r} in FS_METHOD_SETS.")

    if val13_root is None:
        val13_root = (
            REPO_ROOT
            / "run_artifacts"
            / "validation-13"
            / "val13_4hosts_live_20260228_133809"
        )
    val13_runtime = _load_runtime_hints_from_summaries(val13_root, phase_tag="val13")
    val13_a_hist = dict(val13_runtime.get("a_control") or {})
    val13_d_hist = dict(val13_runtime.get("d_default") or {})
    val13_v13_hist = dict(val13_runtime.get("d_v13") or {})

    a_est: Dict[str, float] = {}
    d_default_est: Dict[str, float] = {}
    v14_ref_est: Dict[str, float] = {}
    v14_no_bh_est: Dict[str, float] = {}
    v14_no_varfloor_est: Dict[str, float] = {}
    v14_no_copula5_est: Dict[str, float] = {}
    v14_add_stability_est: Dict[str, float] = {}
    v14_add_pareto_est: Dict[str, float] = {}
    v14_no_extreme_multiclass_est: Dict[str, float] = {}
    v14_tritrust_est: Dict[str, float] = {}
    v14_tabpfn_est: Dict[str, float] = {}
    v14_ipss_est: Dict[str, float] = {}
    v14_ipss_eats_est: Dict[str, float] = {}
    v14_group_sparse_est: Dict[str, float] = {}
    v14_batch_combat_est: Dict[str, float] = {}
    v14_mapie_aps_est: Dict[str, float] = {}
    v14_mapie_raps_est: Dict[str, float] = {}
    v14_mapie_cross_est: Dict[str, float] = {}
    v14_multiomics_adapter_est: Dict[str, float] = {}
    v14_gates_only_est: Dict[str, float] = {}

    for ds_id in validation_ids:
        spec = benchmark_datasets.get(ds_id)
        base_t = float(val13_a_hist.get(ds_id, 0.0))
        if base_t <= 0.0:
            base_t = float(max(60.0, _dataset_weight(spec) * 220.0))
        a_est[ds_id] = float(base_t)

        d_seed = float(val13_d_hist.get(ds_id, 0.0))
        if d_seed <= 0.0:
            d_seed = float(val13_v13_hist.get(ds_id, 0.0))
        if d_seed <= 0.0:
            d_seed = float(max(base_t * 1.35, _dataset_weight(spec) * 260.0))
        d_default_est[ds_id] = float(d_seed)

        ref = float(max(base_t * 1.15, d_seed * 1.06))
        v14_ref_est[ds_id] = ref
        v14_no_bh_est[ds_id] = float(max(base_t * 1.10, ref * 0.98))
        v14_no_varfloor_est[ds_id] = float(max(base_t * 1.08, ref * 0.97))
        v14_no_copula5_est[ds_id] = float(max(base_t * 1.05, ref * 0.95))
        v14_add_stability_est[ds_id] = float(max(base_t * 1.12, ref * 1.06))
        v14_add_pareto_est[ds_id] = float(max(base_t * 1.16, ref * 1.10))
        v14_no_extreme_multiclass_est[ds_id] = float(max(base_t * 1.10, ref * 1.00))
        v14_tritrust_est[ds_id] = float(max(base_t * 1.12, ref * 1.02))
        v14_tabpfn_est[ds_id] = float(max(base_t * 1.18, ref * 1.12))
        v14_ipss_est[ds_id] = float(max(base_t * 1.16, ref * 1.08))
        v14_ipss_eats_est[ds_id] = float(max(base_t * 1.20, v14_ipss_est[ds_id] * 1.04))
        v14_group_sparse_est[ds_id] = float(max(base_t * 1.15, ref * 1.07))
        v14_batch_combat_est[ds_id] = float(max(base_t * 1.18, ref * 1.12))
        v14_mapie_aps_est[ds_id] = float(max(base_t * 1.12, ref * 1.03))
        v14_mapie_raps_est[ds_id] = float(max(base_t * 1.13, ref * 1.04))
        v14_mapie_cross_est[ds_id] = float(max(base_t * 1.18, ref * 1.12))
        v14_multiomics_adapter_est[ds_id] = float(max(base_t * 1.22, ref * 1.16))
        v14_gates_only_est[ds_id] = float(max(base_t * 1.00, ref * 0.92))

    runtime_by_profile = {
        "a_control": a_est,
        "d_default": d_default_est,
        "v14_ref": v14_ref_est,
        "v14_no_bh": v14_no_bh_est,
        "v14_no_varfloor": v14_no_varfloor_est,
        "v14_no_copula5": v14_no_copula5_est,
        "v14_add_stability": v14_add_stability_est,
        "v14_add_pareto": v14_add_pareto_est,
        "v14_no_extreme_multiclass": v14_no_extreme_multiclass_est,
        "v14_tritrust": v14_tritrust_est,
        "v14_tabpfn": v14_tabpfn_est,
        "v14_ipss": v14_ipss_est,
        "v14_ipss_eats": v14_ipss_eats_est,
        "v14_group_sparse": v14_group_sparse_est,
        "v14_batch_combat": v14_batch_combat_est,
        "v14_mapie_aps": v14_mapie_aps_est,
        "v14_mapie_raps": v14_mapie_raps_est,
        "v14_mapie_cross": v14_mapie_cross_est,
        "v14_multiomics_adapter": v14_multiomics_adapter_est,
        "v14_gates_only": v14_gates_only_est,
    }

    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        combined_seed_sec = float(sum(float(rt.get(ds_id, 180.0)) for rt in runtime_by_profile.values()))
        ds_items.append((str(ds_id), combined_seed_sec * float(len(seeds))))
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_extra_args: Tuple[str, ...] = ("--emit-summary", "--compute-budget", "standard")
    legacy_models: Tuple[str, ...] = (
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
    )
    shared_stage_flags: Tuple[str, ...] = (
        "--df-family-set", "flex",
        "--df-compute-ad",
        "--df-compute-qq-pp",
        "--df-compute-dip",
        "--df-interval-likelihood",
        "--df-compute-crps",
        "--df-crps-uq-decomposition",
        "--df-lmoment-prescreen",
        "--df-lmoment-prescreen-max-candidates", "12",
        "--folding-method", "pls_da",
        "--enable-prefilter-rnaseq-nb-lrt",
        "--prefilter-rnaseq-nb-lrt-alpha", "0.10",
        "--enable-classifier-conformal",
        "--classifier-conformal-alpha", "0.10",
        "--classifier-conformal-calibration-fraction", "0.25",
        "--classifier-conformal-min-calibration", "20",
        "--enable-stage2-ratio-augmentation",
        "--stage2-ratio-max-features", "16",
        "--stage2-ratio-selection-method", "correlation",
        "--enable-model-cv-runtime-containment",
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
    )

    a_control_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--mnpo-performance-oracle-mode", "single",
        "--fs-oracle-weighting-mode", "uniform",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        *legacy_models,
        "--model-cv-runtime-max-candidates", "8",
        "--fs-copula-derandomize-runs", "3",
    )

    d_default_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--prefilter-bh-ttest-enabled",
        "--prefilter-variance-floor-enabled",
        "--fs-copula-derandomize-runs", "3",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        *legacy_models,
        "--model-cv-runtime-max-candidates", "8",
    )

    v14_ref_flags: Tuple[str, ...] = (
        *d_default_flags,
        "--regime-gating-enabled",
        "--regime-gating-difficulty-source", "historical",
        "--regime-gating-target-tier", "very_hard",
        "--regime-gating-min-samples-per-class", "7",
        "--regime-gating-very-hard-min-classes", "5",
        "--regime-gating-low-p-over-n-threshold", "2.0",
        "--regime-gating-simple-fs-method-set", "strict_plus_mrmr",
        "--regime-gating-very-hard-portfolio-max-methods", "4",
        "--regime-gating-very-hard-copula-derandomize-runs", "5",
        "--regime-gating-low-p-over-n-mode", "fast_univariate_filter",
        "--regime-gating-extreme-multiclass-enabled",
        "--regime-gating-extreme-multiclass-threshold", "8",
        "--regime-gating-extreme-multiclass-min-samples-per-class", "11",
        "--fs-copula-derandomize-runs", "5",
        "--fs-max-selected-features-ratio", "0.5",
        "--fs-max-selected-features-cap", "500",
        "--fs-stability-threshold-method", "fixed",
    )

    v14_no_bh_flags: Tuple[str, ...] = (*v14_ref_flags, "--no-prefilter-bh-ttest")
    v14_no_varfloor_flags: Tuple[str, ...] = (*v14_ref_flags, "--no-prefilter-variance-floor")
    v14_no_copula5_flags: Tuple[str, ...] = (*v14_ref_flags, "--fs-copula-derandomize-runs", "3")
    v14_add_stability_flags: Tuple[str, ...] = (*v14_ref_flags, "--fs-stability-weighted-aggregation-enabled")
    v14_add_pareto_flags: Tuple[str, ...] = (*v14_ref_flags, "--enable-fs-pareto-portfolio-sizing")
    v14_no_extreme_multiclass_flags: Tuple[str, ...] = (
        *v14_ref_flags,
        "--no-regime-gating-extreme-multiclass",
    )
    v14_tritrust_flags: Tuple[str, ...] = (*v14_ref_flags, "--fs-oracle-weighting-mode", "tritrust")
    v14_tabpfn_flags: Tuple[str, ...] = (
        *v14_ref_flags,
        "--model-candidates",
        *legacy_models,
        "tabpfn",
    )
    v14_ipss_flags: Tuple[str, ...] = tuple(v14_ref_flags)
    v14_ipss_eats_flags: Tuple[str, ...] = (
        *v14_ref_flags,
        "--enable-fs-ipss-eats-threshold",
        "--fs-stability-threshold-method", "eats",
    )
    v14_group_sparse_flags: Tuple[str, ...] = tuple(v14_ref_flags)
    v14_batch_combat_flags: Tuple[str, ...] = (
        *v14_ref_flags,
        "--batch-correction", "combat",
        "--batch-label-policy", "kmeans2",
    )
    v14_mapie_aps_flags: Tuple[str, ...] = (*v14_ref_flags, "--classifier-conformal-method", "aps")
    v14_mapie_raps_flags: Tuple[str, ...] = (*v14_ref_flags, "--classifier-conformal-method", "raps")
    v14_mapie_cross_flags: Tuple[str, ...] = (*v14_ref_flags, "--classifier-conformal-method", "cross")
    v14_multiomics_adapter_flags: Tuple[str, ...] = (
        *v14_ref_flags,
        "--multiomics-adapter", "split_halves",
        "--multiomics-integrator", "mb_plsda",
        "--multiomics-n-components", "2",
    )
    v14_gates_only_flags: Tuple[str, ...] = (
        *v14_ref_flags,
        "--no-prefilter-bh-ttest",
        "--no-prefilter-variance-floor",
        "--fs-copula-derandomize-runs", "3",
    )

    common_job_params = dict(
        seeds=[int(s) for s in seeds],
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        progress_stall_watchdog_sec=1800,
        task_timeout_sec=int(task_timeout_sec),
        fs_method_timeout_sec=int(fs_method_timeout_sec),
    )

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile("a_control", "strict_plus_mrmr", a_control_flags, notes="Val-14 legacy simple anchor."),
        BenchmarkProfile("d_default", "mnpo_broad_all", d_default_flags, notes="Val-14 production anchor."),
        BenchmarkProfile("v14_ref", "mnpo_v14_core", v14_ref_flags, notes="Val-14 corrected reference profile."),
        BenchmarkProfile("v14_no_bh", "mnpo_v14_core", v14_no_bh_flags, notes="Val-14 ablation: disable BH prefilter."),
        BenchmarkProfile(
            "v14_no_varfloor",
            "mnpo_v14_core",
            v14_no_varfloor_flags,
            notes="Val-14 ablation: disable variance floor.",
        ),
        BenchmarkProfile(
            "v14_no_copula5",
            "mnpo_v14_core",
            v14_no_copula5_flags,
            notes="Val-14 ablation: copula derandomize runs 5->3.",
        ),
        BenchmarkProfile(
            "v14_add_stability",
            "mnpo_v14_core",
            v14_add_stability_flags,
            notes="Val-14 additive: enable stability-weighted aggregation.",
        ),
        BenchmarkProfile(
            "v14_add_pareto",
            "mnpo_v14_core",
            v14_add_pareto_flags,
            notes="Val-14 additive: enable Pareto portfolio sizing.",
        ),
        BenchmarkProfile(
            "v14_no_extreme_multiclass",
            "mnpo_v14_core",
            v14_no_extreme_multiclass_flags,
            notes="Val-14 ablation: disable extreme multiclass gate.",
        ),
        BenchmarkProfile(
            "v14_tritrust",
            "mnpo_v14_core",
            v14_tritrust_flags,
            notes="Val-14 ablation: oracle weighting banzhaf->tritrust.",
        ),
        BenchmarkProfile(
            "v14_tabpfn",
            "mnpo_v14_core",
            v14_tabpfn_flags,
            notes="Val-14 additive: include TabPFN classifier candidate.",
        ),
        BenchmarkProfile("v14_ipss", "mnpo_v14_core_plus_ipss", v14_ipss_flags, notes="Val-14 additive: include IPSS."),
        BenchmarkProfile(
            "v14_ipss_eats",
            "mnpo_v14_core_plus_ipss",
            v14_ipss_eats_flags,
            notes="Val-14 additive: IPSS + EATS thresholding.",
        ),
        BenchmarkProfile(
            "v14_group_sparse",
            "mnpo_v14_core_plus_group_sparse",
            v14_group_sparse_flags,
            notes="Val-14 additive: include group_sparse_lasso.",
        ),
        BenchmarkProfile(
            "v14_batch_combat",
            "mnpo_v14_core",
            v14_batch_combat_flags,
            notes="Val-14 additive: ComBat batch correction with kmeans2 batch labels.",
        ),
        BenchmarkProfile("v14_mapie_aps", "mnpo_v14_core", v14_mapie_aps_flags, notes="Val-14 additive: MAPIE APS."),
        BenchmarkProfile("v14_mapie_raps", "mnpo_v14_core", v14_mapie_raps_flags, notes="Val-14 additive: MAPIE RAPS."),
        BenchmarkProfile(
            "v14_mapie_cross",
            "mnpo_v14_core",
            v14_mapie_cross_flags,
            notes="Val-14 additive: MAPIE cross-conformal.",
        ),
        BenchmarkProfile(
            "v14_multiomics_adapter",
            "mnpo_v14_core",
            v14_multiomics_adapter_flags,
            notes="Val-14 additive: benchmark-time split-halves multi-omics adapter.",
        ),
        BenchmarkProfile(
            "v14_gates_only",
            "mnpo_v14_core",
            v14_gates_only_flags,
            notes="Val-14 bundle ablation: gates only (BH/VF off, copula=3).",
        ),
    ]

    _assert_no_implicit_true_omissions(
        profiles=profiles,
        required_negations_by_profile={
            "v14_no_bh": ("--no-prefilter-bh-ttest",),
            "v14_no_varfloor": ("--no-prefilter-variance-floor",),
            "v14_no_extreme_multiclass": ("--no-regime-gating-extreme-multiclass",),
            "v14_gates_only": ("--no-prefilter-bh-ttest", "--no-prefilter-variance-floor"),
        },
        required_value_overrides_by_profile={
            "v14_ref": (("--fs-copula-derandomize-runs", "5"),),
            "v14_no_copula5": (("--fs-copula-derandomize-runs", "3"),),
            "v14_gates_only": (("--fs-copula-derandomize-runs", "3"),),
        },
        context=phase_prefix,
    )

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError(f"Invalid {phase_prefix} profile(s):\n- " + "\n- ".join(invalid))

    jobs: List[Job] = []
    for prof in profiles:
        profile_runtime = dict(runtime_by_profile.get(prof.profile_id) or {})
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_seed_sec = 0.0
            for ds_id in ds_list:
                part_seed_sec += float(profile_runtime.get(ds_id, 180.0))
            part_weight = float(part_seed_sec * len(seeds))
            jobs.append(
                _job(
                    f"{phase_prefix}/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    profile_notes=str(prof.notes),
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    **common_job_params,
                )
            )
    return jobs


def build_jobs_validation14(
    *,
    dataset_shards: int = 6,
    val13_root: Optional[Path] = None,
) -> List[Job]:
    """Validation-14 full matrix (20 profiles x 35 datasets x 9 seeds)."""
    return _build_jobs_validation14_common(
        dataset_ids=list(VALIDATION14_DATASETS),
        dataset_shards=int(max(1, dataset_shards)),
        val13_root=val13_root,
        seeds=list(VALIDATION_SEEDS),
        phase_prefix="val14",
        task_timeout_sec=21600,
        fs_method_timeout_sec=3600,
    )


def build_jobs_validation14_activation_smoke(
    *,
    dataset_shards: int = 6,
    val13_root: Optional[Path] = None,
) -> List[Job]:
    """Validation-14 activation smoke (same profiles, 6 datasets, single seed)."""
    return _build_jobs_validation14_common(
        dataset_ids=list(VALIDATION14_ACTIVATION_DATASETS),
        dataset_shards=int(max(1, dataset_shards)),
        val13_root=val13_root,
        seeds=[11],
        phase_prefix="val14_activation",
        task_timeout_sec=7200,
        fs_method_timeout_sec=1800,
    )


def _balanced_shard_assign_validation14_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-14 by dataset partitions (group all profile bundles per partition)."""
    profile_re = re.compile(
        r"^val14/(" + "|".join(re.escape(p) for p in sorted(VALIDATION14_PROFILE_MANIFEST.keys())) + r")/(ds\d+)$"
    )
    grouped: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = profile_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        grouped.setdefault(part_id, {})[profile_id] = job

    bundle_items: List[Tuple[float, List[Job]]] = []
    for part_id, bundle in grouped.items():
        bundle_weight = sum(float(j.weight) for j in bundle.values())
        bundle_items.append((bundle_weight, list(bundle.values())))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for bundle_weight, bundle_jobs in sorted(bundle_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        for j in sorted(bundle_jobs, key=lambda j: j.job_id):
            shards[target].append(j.job_id)
        totals[target] += float(bundle_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


def _balanced_shard_assign_validation14_activation_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-14 activation smoke by dataset partition bundles."""
    profile_re = re.compile(
        r"^val14_activation/(" + "|".join(re.escape(p) for p in sorted(VALIDATION14_PROFILE_MANIFEST.keys())) + r")/(ds\d+)$"
    )
    grouped: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = profile_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        grouped.setdefault(part_id, {})[profile_id] = job

    bundle_items: List[Tuple[float, List[Job]]] = []
    for part_id, bundle in grouped.items():
        bundle_weight = sum(float(j.weight) for j in bundle.values())
        bundle_items.append((bundle_weight, list(bundle.values())))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for bundle_weight, bundle_jobs in sorted(bundle_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        for j in sorted(bundle_jobs, key=lambda j: j.job_id):
            shards[target].append(j.job_id)
        totals[target] += float(bundle_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


# ---------------------------------------------------------------------------
# Validation-15 focused matrix (post-Val-14 course corrections)
# ---------------------------------------------------------------------------

VALIDATION15_DATASETS: List[str] = list(VALIDATION14_DATASETS)

VALIDATION15_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "a_control": {"anchor": "yes", "contrast_ref": "v15_ref_ipss", "effect": "legacy_simple_anchor"},
    "d_default": {"anchor": "yes", "contrast_ref": "v15_ref_ipss", "effect": "production_anchor"},
    "v15_ref_ipss": {"anchor": "yes", "contrast_ref": "d_default", "effect": "corrected_reference_ipss"},
    "v15_no_ipss": {"anchor": "no", "contrast_ref": "v15_ref_ipss", "effect": "ipss_ablation"},
    "v15_no_regime_fallback": {"anchor": "no", "contrast_ref": "v15_ref_ipss", "effect": "very_hard_fallback_ablation"},
    "v15_mapie_aps": {"anchor": "no", "contrast_ref": "v15_ref_ipss", "effect": "mapie_aps"},
    "v15_mapie_raps": {"anchor": "no", "contrast_ref": "v15_ref_ipss", "effect": "mapie_raps"},
    "v15_mapie_cross": {"anchor": "no", "contrast_ref": "v15_ref_ipss", "effect": "mapie_cross"},
    "v15_multiomics_adapter": {"anchor": "no", "contrast_ref": "v15_ref_ipss", "effect": "multiomics_adapter"},
}


def _build_jobs_validation15_common(
    *,
    dataset_ids: Sequence[str],
    dataset_shards: int,
    val14_root: Optional[Path],
    seeds: Sequence[int],
    phase_prefix: str,
    task_timeout_sec: int,
    fs_method_timeout_sec: int,
) -> List[Job]:
    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, _dataset_sets = _load_benchmark_registry()
    hf_meta = _load_hf_manifest_metadata()
    hf_ids = set(hf_meta.keys())
    if not hf_ids:
        raise RuntimeError("HF bundle manifest metadata is empty; cannot build validation15 plan.")

    validation_ids = [str(ds) for ds in dataset_ids if str(ds) in hf_ids]
    if not validation_ids:
        raise RuntimeError(f"No datasets are available in HF bundle manifests for {phase_prefix}.")

    missing_hf = [str(ds) for ds in dataset_ids if str(ds) not in hf_ids]
    if missing_hf:
        preview = ", ".join(missing_hf[:10])
        suffix = " ..." if len(missing_hf) > 10 else ""
        print(
            f"[{phase_prefix}] Skipping {len(missing_hf)} dataset(s) not present in "
            f"HF manifests: {preview}{suffix}",
            file=sys.stderr,
        )

    required_method_sets = (
        "strict_plus_mrmr",
        "mnpo_broad_all",
        "mnpo_v14_core",
        "mnpo_v14_core_plus_ipss",
    )
    for required in required_method_sets:
        if required not in fs_method_sets:
            raise RuntimeError(f"Missing required method set {required!r} in FS_METHOD_SETS.")

    if val14_root is None:
        val14_root = (
            REPO_ROOT
            / "run_artifacts"
            / "validation-14"
            / "val14_3hosts_live_20260301_223239"
        )
    val14_runtime = _load_runtime_hints_from_summaries(val14_root, phase_tag="val14")
    val14_a_hist = dict(val14_runtime.get("a_control") or {})
    val14_d_hist = dict(val14_runtime.get("d_default") or {})
    val14_ref_hist = dict(val14_runtime.get("v14_ref") or {})
    val14_ipss_hist = dict(val14_runtime.get("v14_ipss") or {})
    val14_multiomics_hist = dict(val14_runtime.get("v14_multiomics_adapter") or {})

    a_est: Dict[str, float] = {}
    d_default_est: Dict[str, float] = {}
    v15_ref_ipss_est: Dict[str, float] = {}
    v15_no_ipss_est: Dict[str, float] = {}
    v15_no_regime_fallback_est: Dict[str, float] = {}
    v15_mapie_aps_est: Dict[str, float] = {}
    v15_mapie_raps_est: Dict[str, float] = {}
    v15_mapie_cross_est: Dict[str, float] = {}
    v15_multiomics_adapter_est: Dict[str, float] = {}

    for ds_id in validation_ids:
        spec = benchmark_datasets.get(ds_id)
        base_t = float(val14_a_hist.get(ds_id, 0.0))
        if base_t <= 0.0:
            base_t = float(max(60.0, _dataset_weight(spec) * 220.0))
        a_est[ds_id] = float(base_t)

        d_seed = float(val14_d_hist.get(ds_id, 0.0))
        if d_seed <= 0.0:
            d_seed = float(val14_ref_hist.get(ds_id, 0.0))
        if d_seed <= 0.0:
            d_seed = float(max(base_t * 1.30, _dataset_weight(spec) * 260.0))
        d_default_est[ds_id] = float(d_seed)

        ref_seed = float(val14_ipss_hist.get(ds_id, 0.0))
        if ref_seed <= 0.0:
            ref_seed = float(val14_ref_hist.get(ds_id, 0.0))
        if ref_seed <= 0.0:
            ref_seed = float(max(base_t * 1.14, d_seed * 1.04))
        v15_ref_ipss_est[ds_id] = float(ref_seed)

        v15_no_ipss_est[ds_id] = float(max(base_t * 1.10, ref_seed * 0.98))
        v15_no_regime_fallback_est[ds_id] = float(max(base_t * 1.08, ref_seed * 0.97))
        v15_mapie_aps_est[ds_id] = float(max(base_t * 1.12, ref_seed * 1.03))
        v15_mapie_raps_est[ds_id] = float(max(base_t * 1.13, ref_seed * 1.04))
        v15_mapie_cross_est[ds_id] = float(max(base_t * 1.18, ref_seed * 1.12))

        mo_seed = float(val14_multiomics_hist.get(ds_id, 0.0))
        if mo_seed <= 0.0:
            mo_seed = float(max(base_t * 1.22, ref_seed * 1.16))
        v15_multiomics_adapter_est[ds_id] = float(mo_seed)

    runtime_by_profile = {
        "a_control": a_est,
        "d_default": d_default_est,
        "v15_ref_ipss": v15_ref_ipss_est,
        "v15_no_ipss": v15_no_ipss_est,
        "v15_no_regime_fallback": v15_no_regime_fallback_est,
        "v15_mapie_aps": v15_mapie_aps_est,
        "v15_mapie_raps": v15_mapie_raps_est,
        "v15_mapie_cross": v15_mapie_cross_est,
        "v15_multiomics_adapter": v15_multiomics_adapter_est,
    }

    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        combined_seed_sec = float(sum(float(rt.get(ds_id, 180.0)) for rt in runtime_by_profile.values()))
        ds_items.append((str(ds_id), combined_seed_sec * float(len(seeds))))
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_extra_args: Tuple[str, ...] = ("--emit-summary", "--compute-budget", "standard")
    legacy_models: Tuple[str, ...] = (
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
    )
    shared_stage_flags: Tuple[str, ...] = (
        "--df-family-set", "flex",
        "--df-compute-ad",
        "--df-compute-qq-pp",
        "--df-compute-dip",
        "--df-interval-likelihood",
        "--df-compute-crps",
        "--df-crps-uq-decomposition",
        "--df-lmoment-prescreen",
        "--df-lmoment-prescreen-max-candidates", "12",
        "--folding-method", "pls_da",
        "--enable-prefilter-rnaseq-nb-lrt",
        "--prefilter-rnaseq-nb-lrt-alpha", "0.10",
        "--enable-classifier-conformal",
        "--classifier-conformal-alpha", "0.10",
        "--classifier-conformal-calibration-fraction", "0.25",
        "--classifier-conformal-min-calibration", "20",
        "--enable-stage2-ratio-augmentation",
        "--stage2-ratio-max-features", "16",
        "--stage2-ratio-selection-method", "correlation",
        "--enable-model-cv-runtime-containment",
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
    )

    a_control_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--mnpo-performance-oracle-mode", "single",
        "--fs-oracle-weighting-mode", "uniform",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        *legacy_models,
        "--model-cv-runtime-max-candidates", "8",
        "--fs-copula-derandomize-runs", "3",
    )

    d_default_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--prefilter-bh-ttest-enabled",
        "--prefilter-variance-floor-enabled",
        "--fs-copula-derandomize-runs", "3",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        *legacy_models,
        "--model-cv-runtime-max-candidates", "8",
    )

    v15_ref_ipss_flags: Tuple[str, ...] = (
        *d_default_flags,
        "--regime-gating-enabled",
        "--regime-gating-difficulty-source", "historical",
        "--regime-gating-target-tier", "very_hard",
        "--regime-gating-min-samples-per-class", "7",
        "--regime-gating-very-hard-min-classes", "5",
        "--regime-gating-low-p-over-n-threshold", "0",
        "--regime-gating-simple-fs-method-set", "strict_plus_mrmr",
        "--regime-gating-very-hard-portfolio-max-methods", "4",
        "--regime-gating-very-hard-copula-derandomize-runs", "5",
        "--regime-gating-extreme-multiclass-enabled",
        "--regime-gating-extreme-multiclass-threshold", "8",
        "--regime-gating-extreme-multiclass-min-samples-per-class", "11",
        "--fs-copula-derandomize-runs", "5",
        "--fs-max-selected-features-ratio", "0.5",
        "--fs-max-selected-features-cap", "500",
        "--fs-stability-threshold-method", "fixed",
    )

    v15_no_ipss_flags: Tuple[str, ...] = tuple(v15_ref_ipss_flags)
    v15_no_regime_fallback_flags: Tuple[str, ...] = (
        *d_default_flags,
        "--fs-copula-derandomize-runs", "5",
        "--fs-max-selected-features-ratio", "0.5",
        "--fs-max-selected-features-cap", "500",
        "--fs-stability-threshold-method", "fixed",
    )
    v15_mapie_aps_flags: Tuple[str, ...] = (*v15_ref_ipss_flags, "--classifier-conformal-method", "aps")
    v15_mapie_raps_flags: Tuple[str, ...] = (*v15_ref_ipss_flags, "--classifier-conformal-method", "raps")
    v15_mapie_cross_flags: Tuple[str, ...] = (*v15_ref_ipss_flags, "--classifier-conformal-method", "cross")
    v15_multiomics_adapter_flags: Tuple[str, ...] = (
        *v15_ref_ipss_flags,
        "--multiomics-adapter", "split_halves",
        "--multiomics-integrator", "mb_plsda",
        "--multiomics-n-components", "2",
    )

    common_job_params = dict(
        seeds=[int(s) for s in seeds],
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        progress_stall_watchdog_sec=1800,
        task_timeout_sec=int(task_timeout_sec),
        fs_method_timeout_sec=int(fs_method_timeout_sec),
    )

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile("a_control", "strict_plus_mrmr", a_control_flags, notes="Val-15 legacy simple anchor."),
        BenchmarkProfile("d_default", "mnpo_broad_all", d_default_flags, notes="Val-15 production anchor."),
        BenchmarkProfile(
            "v15_ref_ipss",
            "mnpo_v14_core_plus_ipss",
            v15_ref_ipss_flags,
            notes=(
                "Val-15 corrected reference: IPSS in core stack, very-hard fallback retained, "
                "low-p/n bypass disabled."
            ),
        ),
        BenchmarkProfile(
            "v15_no_ipss",
            "mnpo_v14_core",
            v15_no_ipss_flags,
            notes="Val-15 ablation: remove IPSS from corrected reference.",
        ),
        BenchmarkProfile(
            "v15_no_regime_fallback",
            "mnpo_v14_core_plus_ipss",
            v15_no_regime_fallback_flags,
            notes="Val-15 ablation: disable regime gating to isolate very-hard fallback value.",
        ),
        BenchmarkProfile("v15_mapie_aps", "mnpo_v14_core_plus_ipss", v15_mapie_aps_flags, notes="Val-15 additive: MAPIE APS."),
        BenchmarkProfile("v15_mapie_raps", "mnpo_v14_core_plus_ipss", v15_mapie_raps_flags, notes="Val-15 additive: MAPIE RAPS."),
        BenchmarkProfile(
            "v15_mapie_cross",
            "mnpo_v14_core_plus_ipss",
            v15_mapie_cross_flags,
            notes="Val-15 additive: MAPIE cross-conformal.",
        ),
        BenchmarkProfile(
            "v15_multiomics_adapter",
            "mnpo_v14_core_plus_ipss",
            v15_multiomics_adapter_flags,
            notes="Val-15 mechanism probe: split-halves multi-omics adapter.",
        ),
    ]

    _assert_no_implicit_true_omissions(
        profiles=profiles,
        required_value_overrides_by_profile={
            "v15_ref_ipss": (("--regime-gating-low-p-over-n-threshold", "0"),),
            "v15_mapie_aps": (("--classifier-conformal-method", "aps"),),
            "v15_mapie_raps": (("--classifier-conformal-method", "raps"),),
            "v15_mapie_cross": (("--classifier-conformal-method", "cross"),),
        },
        context=phase_prefix,
    )

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError(f"Invalid {phase_prefix} profile(s):\n- " + "\n- ".join(invalid))

    jobs: List[Job] = []
    for prof in profiles:
        profile_runtime = dict(runtime_by_profile.get(prof.profile_id) or {})
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_seed_sec = 0.0
            for ds_id in ds_list:
                part_seed_sec += float(profile_runtime.get(ds_id, 180.0))
            part_weight = float(part_seed_sec * len(seeds))
            jobs.append(
                _job(
                    f"{phase_prefix}/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    profile_notes=str(prof.notes),
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    **common_job_params,
                )
            )
    return jobs


def build_jobs_validation15(
    *,
    dataset_shards: int = 6,
    val14_root: Optional[Path] = None,
) -> List[Job]:
    """Validation-15 focused matrix (9 profiles x 35 datasets x 9 seeds)."""
    return _build_jobs_validation15_common(
        dataset_ids=list(VALIDATION15_DATASETS),
        dataset_shards=int(max(1, dataset_shards)),
        val14_root=val14_root,
        seeds=list(VALIDATION_SEEDS),
        phase_prefix="val15",
        task_timeout_sec=21600,
        fs_method_timeout_sec=3600,
    )


def _balanced_shard_assign_validation15_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-15 by dataset partitions (group all profile bundles per partition)."""
    profile_re = re.compile(
        r"^val15/(" + "|".join(re.escape(p) for p in sorted(VALIDATION15_PROFILE_MANIFEST.keys())) + r")/(ds\d+)$"
    )
    grouped: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = profile_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        grouped.setdefault(part_id, {})[profile_id] = job

    bundle_items: List[Tuple[float, List[Job]]] = []
    for part_id, bundle in grouped.items():
        bundle_weight = sum(float(j.weight) for j in bundle.values())
        bundle_items.append((bundle_weight, list(bundle.values())))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for bundle_weight, bundle_jobs in sorted(bundle_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        for j in sorted(bundle_jobs, key=lambda j: j.job_id):
            shards[target].append(j.job_id)
        totals[target] += float(bundle_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


# ---------------------------------------------------------------------------
# Validation-16 focused matrix (post-Val-15 oracle stabilization / meta-learning)
# ---------------------------------------------------------------------------

VALIDATION16_DATASETS: List[str] = list(VALIDATION15_DATASETS) + [
    "mll_microarray",
    "smk_can_187",
    "gla_bra_180",
    "brain_tumor_2_50_4class",
    "orlraws10p",
    "warp_pie10p",
    "pixraw10p",
    "lymphoma_11",
    "gisette_nips03",
    "dexter_nips03",
    "lung_gordon",
    "nci60_ross",
    "cumida_breast_gse45827",
    "cumida_prostate_gse6919",
    "cumida_ovarian_gse26712",
    "cumida_lung_gse19804",
    "cumida_pancreatic_gse16515",
    "cumida_headneck_gse12452",
    "xena_tcga_brca",
    "xena_tcga_luad",
    "xena_tcga_ucec",
    "xena_tcga_kirc",
    "xena_tcga_hnsc_hpv",
    "xena_tcga_stad",
    "xena_tcga_lihc",
    "xena_tcga_ov",
    "xena_tcga_prad",
    "rv_coil20",
    "rv_basehock",
]

VALIDATION16_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "a_control": {"anchor": "yes", "contrast_ref": "v16_ref", "effect": "legacy_simple_anchor"},
    "d_default": {"anchor": "yes", "contrast_ref": "v16_ref", "effect": "production_anchor"},
    "v16_ref": {"anchor": "yes", "contrast_ref": "d_default", "effect": "corrected_reference_ipss"},
    "v16_clp": {"anchor": "no", "contrast_ref": "v16_ref", "effect": "continuous_logistic_preferences"},
    "v16_payoff_shrink": {"anchor": "no", "contrast_ref": "v16_ref", "effect": "payoff_shrinkage"},
    "v16_clp_shrink": {"anchor": "no", "contrast_ref": "v16_ref", "effect": "clp_plus_payoff_shrinkage"},
    "v16_conformal_eff": {"anchor": "no", "contrast_ref": "v16_ref", "effect": "conformal_efficiency_aps"},
    "v16_js_shrinkage": {"anchor": "no", "contrast_ref": "v16_ref", "effect": "oracle_weight_js_shrinkage"},
    "v16_meta_dt": {"anchor": "no", "contrast_ref": "v16_ref", "effect": "meta_learning_decision_tree"},
    "v16_multiomics": {"anchor": "no", "contrast_ref": "v16_ref", "effect": "multiomics_adapter"},
    "v16_full_stack": {"anchor": "no", "contrast_ref": "v16_ref", "effect": "full_val16_stack"},
}


VALIDATION17_DATASETS: List[str] = list(VALIDATION16_DATASETS)

VALIDATION17_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = dict(VALIDATION16_PROFILE_MANIFEST)


def _build_jobs_validation16_family(
    *,
    run_family: str,
    validation_label: str,
    validation_datasets: Sequence[str],
    profile_note_prefix: str,
    dataset_integrity_policy: str,
    common_extra_args: Sequence[str] = (),
    dataset_shards: int = 9,
    val15_root: Optional[Path] = None,
) -> List[Job]:
    """Shared Val-16/Val-17 focused matrix builder."""
    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, _dataset_sets = _load_benchmark_registry()
    hf_meta = _load_hf_manifest_metadata()
    hf_ids = set(hf_meta.keys())
    if not hf_ids:
        raise RuntimeError(f"HF bundle manifest metadata is empty; cannot build {validation_label} plan.")

    validation_ids = [str(ds) for ds in validation_datasets if str(ds) in hf_ids]
    if not validation_ids:
        raise RuntimeError(f"No datasets are available in HF bundle manifests for {validation_label}.")

    missing_hf = [str(ds) for ds in validation_datasets if str(ds) not in hf_ids]
    if missing_hf:
        preview = ", ".join(missing_hf[:10])
        suffix = " ..." if len(missing_hf) > 10 else ""
        print(
            f"[{validation_label}] Skipping {len(missing_hf)} dataset(s) not present in "
            f"HF manifests: {preview}{suffix}",
            file=sys.stderr,
        )

    required_method_sets = (
        "strict_plus_mrmr",
        "mnpo_broad_all",
        "mnpo_v14_core_plus_ipss",
    )
    for required in required_method_sets:
        if required not in fs_method_sets:
            raise RuntimeError(f"Missing required method set {required!r} in FS_METHOD_SETS.")

    if val15_root is None:
        val15_root = (
            REPO_ROOT
            / "run_artifacts"
            / "validation-15"
            / "val15_rerun_3hosts_merged_20260305_124113"
        )
    val15_runtime = _load_runtime_hints_from_summaries(val15_root, phase_tag="val15")
    val15_a_hist = dict(val15_runtime.get("a_control") or {})
    val15_d_hist = dict(val15_runtime.get("d_default") or {})
    val15_ref_hist = dict(val15_runtime.get("v15_ref_ipss") or {})
    val15_multiomics_hist = dict(val15_runtime.get("v15_multiomics_adapter") or {})

    runtime_by_profile: Dict[str, Dict[str, float]] = {
        "a_control": {},
        "d_default": {},
        "v16_ref": {},
        "v16_clp": {},
        "v16_payoff_shrink": {},
        "v16_clp_shrink": {},
        "v16_conformal_eff": {},
        "v16_js_shrinkage": {},
        "v16_meta_dt": {},
        "v16_multiomics": {},
        "v16_full_stack": {},
    }
    for ds_id in validation_ids:
        spec = benchmark_datasets.get(ds_id)
        base_t = float(val15_a_hist.get(ds_id, 0.0))
        if base_t <= 0.0:
            base_t = float(max(60.0, _dataset_weight(spec) * 220.0))
        default_t = float(val15_d_hist.get(ds_id, 0.0))
        if default_t <= 0.0:
            default_t = float(max(base_t * 1.30, _dataset_weight(spec) * 260.0))
        ref_t = float(val15_ref_hist.get(ds_id, 0.0))
        if ref_t <= 0.0:
            ref_t = float(max(default_t * 1.03, base_t * 1.16))
        multiomics_t = float(val15_multiomics_hist.get(ds_id, 0.0))
        if multiomics_t <= 0.0:
            multiomics_t = float(max(ref_t * 1.16, base_t * 1.22))

        runtime_by_profile["a_control"][ds_id] = float(base_t)
        runtime_by_profile["d_default"][ds_id] = float(default_t)
        runtime_by_profile["v16_ref"][ds_id] = float(ref_t)
        runtime_by_profile["v16_clp"][ds_id] = float(max(base_t * 1.17, ref_t * 1.01))
        runtime_by_profile["v16_payoff_shrink"][ds_id] = float(max(base_t * 1.16, ref_t * 1.00))
        runtime_by_profile["v16_clp_shrink"][ds_id] = float(max(base_t * 1.18, ref_t * 1.02))
        runtime_by_profile["v16_conformal_eff"][ds_id] = float(max(base_t * 1.22, ref_t * 1.05))
        runtime_by_profile["v16_js_shrinkage"][ds_id] = float(max(base_t * 1.16, ref_t * 1.00))
        runtime_by_profile["v16_meta_dt"][ds_id] = float(max(base_t * 1.18, ref_t * 1.03))
        runtime_by_profile["v16_multiomics"][ds_id] = float(multiomics_t)
        runtime_by_profile["v16_full_stack"][ds_id] = float(max(base_t * 1.24, ref_t * 1.08))

    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        combined_seed_sec = float(sum(float(rt.get(ds_id, 180.0)) for rt in runtime_by_profile.values()))
        ds_items.append((str(ds_id), combined_seed_sec * float(len(VALIDATION_SEEDS))))
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    resolved_common_extra_args: Tuple[str, ...] = (
        "--emit-summary",
        "--compute-budget",
        "standard",
        *tuple(common_extra_args),
    )
    legacy_models: Tuple[str, ...] = (
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
    )
    shared_stage_flags: Tuple[str, ...] = (
        "--df-family-set", "flex",
        "--df-compute-ad",
        "--df-compute-qq-pp",
        "--df-compute-dip",
        "--df-interval-likelihood",
        "--df-compute-crps",
        "--df-crps-uq-decomposition",
        "--df-lmoment-prescreen",
        "--df-lmoment-prescreen-max-candidates", "12",
        "--folding-method", "pls_da",
        "--enable-prefilter-rnaseq-nb-lrt",
        "--prefilter-rnaseq-nb-lrt-alpha", "0.10",
        "--enable-classifier-conformal",
        "--classifier-conformal-alpha", "0.10",
        "--classifier-conformal-calibration-fraction", "0.25",
        "--classifier-conformal-min-calibration", "20",
        "--enable-stage2-ratio-augmentation",
        "--stage2-ratio-max-features", "16",
        "--stage2-ratio-selection-method", "correlation",
        "--enable-model-cv-runtime-containment",
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
    )
    a_control_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--mnpo-performance-oracle-mode", "single",
        "--fs-oracle-weighting-mode", "uniform",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        *legacy_models,
        "--model-cv-runtime-max-candidates", "8",
        "--fs-copula-derandomize-runs", "3",
    )
    d_default_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--prefilter-bh-ttest-enabled",
        "--prefilter-variance-floor-enabled",
        "--fs-copula-derandomize-runs", "3",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates",
        *legacy_models,
        "--model-cv-runtime-max-candidates", "8",
    )
    v16_ref_flags: Tuple[str, ...] = (
        *d_default_flags,
        "--regime-gating-enabled",
        "--regime-gating-difficulty-source", "historical",
        "--regime-gating-target-tier", "very_hard",
        "--regime-gating-min-samples-per-class", "7",
        "--regime-gating-very-hard-min-classes", "5",
        "--regime-gating-low-p-over-n-threshold", "0",
        "--regime-gating-simple-fs-method-set", "strict_plus_mrmr",
        "--regime-gating-very-hard-portfolio-max-methods", "4",
        "--regime-gating-very-hard-copula-derandomize-runs", "5",
        "--regime-gating-extreme-multiclass-enabled",
        "--regime-gating-extreme-multiclass-threshold", "8",
        "--regime-gating-extreme-multiclass-min-samples-per-class", "11",
        "--fs-copula-derandomize-runs", "5",
        "--fs-max-selected-features-ratio", "0.5",
        "--fs-max-selected-features-cap", "500",
        "--fs-stability-threshold-method", "fixed",
    )

    profile_flags: Dict[str, Tuple[str, ...]] = {
        "a_control": a_control_flags,
        "d_default": d_default_flags,
        "v16_ref": v16_ref_flags,
        "v16_clp": (*v16_ref_flags, "--fs-fold-preference-mode", "logistic"),
        "v16_payoff_shrink": (*v16_ref_flags, "--fs-payoff-shrinkage-kappa", "0.15"),
        "v16_clp_shrink": (
            *v16_ref_flags,
            "--fs-fold-preference-mode", "logistic",
            "--fs-payoff-shrinkage-kappa", "0.15",
        ),
        "v16_conformal_eff": (
            *v16_ref_flags,
            "--fs-use-conformal-efficiency",
            "--fs-conformal-efficiency-method", "aps",
        ),
        "v16_js_shrinkage": (*v16_ref_flags, "--fs-oracle-weight-js-shrinkage"),
        "v16_meta_dt": (
            *v16_ref_flags,
            "--meta-learning-selector", "decision_tree",
            "--meta-learning-confidence-threshold", "0.55",
        ),
        "v16_multiomics": (
            *v16_ref_flags,
            "--multiomics-adapter", "split_halves",
            "--multiomics-integrator", "mb_plsda",
            "--multiomics-n-components", "2",
        ),
        "v16_full_stack": (
            *v16_ref_flags,
            "--fs-fold-preference-mode", "logistic",
            "--fs-payoff-shrinkage-kappa", "0.15",
            "--fs-use-conformal-efficiency",
            "--fs-conformal-efficiency-method", "aps",
            "--fs-oracle-weight-js-shrinkage",
        ),
    }

    common_job_params = dict(
        seeds=list(VALIDATION_SEEDS),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy=str(dataset_integrity_policy),
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        progress_stall_watchdog_sec=1800,
        task_timeout_sec=21600,
        fs_method_timeout_sec=3600,
    )
    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile("a_control", "strict_plus_mrmr", profile_flags["a_control"], notes=f"{profile_note_prefix} legacy simple anchor."),
        BenchmarkProfile("d_default", "mnpo_broad_all", profile_flags["d_default"], notes=f"{profile_note_prefix} production anchor."),
        BenchmarkProfile("v16_ref", "mnpo_v14_core_plus_ipss", profile_flags["v16_ref"], notes=f"{profile_note_prefix} reference: Val-15 corrected IPSS lineage."),
        BenchmarkProfile("v16_clp", "mnpo_v14_core_plus_ipss", profile_flags["v16_clp"], notes=f"{profile_note_prefix} additive: continuous logistic fold preferences."),
        BenchmarkProfile("v16_payoff_shrink", "mnpo_v14_core_plus_ipss", profile_flags["v16_payoff_shrink"], notes=f"{profile_note_prefix} additive: payoff shrinkage kappa=0.15."),
        BenchmarkProfile("v16_clp_shrink", "mnpo_v14_core_plus_ipss", profile_flags["v16_clp_shrink"], notes=f"{profile_note_prefix} additive: CLP + payoff shrinkage."),
        BenchmarkProfile("v16_conformal_eff", "mnpo_v14_core_plus_ipss", profile_flags["v16_conformal_eff"], notes=f"{profile_note_prefix} additive: conformal efficiency APS oracle."),
        BenchmarkProfile("v16_js_shrinkage", "mnpo_v14_core_plus_ipss", profile_flags["v16_js_shrinkage"], notes=f"{profile_note_prefix} additive: James-Stein oracle-weight shrinkage."),
        BenchmarkProfile("v16_meta_dt", "mnpo_v14_core_plus_ipss", profile_flags["v16_meta_dt"], notes=f"{profile_note_prefix} additive: meta-learning selector routing."),
        BenchmarkProfile("v16_multiomics", "mnpo_v14_core_plus_ipss", profile_flags["v16_multiomics"], notes=f"{profile_note_prefix} additive: split-halves multi-omics adapter."),
        BenchmarkProfile("v16_full_stack", "mnpo_v14_core_plus_ipss", profile_flags["v16_full_stack"], notes=f"{profile_note_prefix} additive: CLP + shrinkage + conformal efficiency + JS."),
    ]

    _assert_no_implicit_true_omissions(
        profiles=profiles,
        required_value_overrides_by_profile={
            "v16_ref": (("--regime-gating-low-p-over-n-threshold", "0"),),
            "v16_clp": (("--fs-fold-preference-mode", "logistic"),),
            "v16_payoff_shrink": (("--fs-payoff-shrinkage-kappa", "0.15"),),
            "v16_clp_shrink": (
                ("--fs-fold-preference-mode", "logistic"),
                ("--fs-payoff-shrinkage-kappa", "0.15"),
            ),
            "v16_conformal_eff": (("--fs-conformal-efficiency-method", "aps"),),
            "v16_meta_dt": (
                ("--meta-learning-selector", "decision_tree"),
                ("--meta-learning-confidence-threshold", "0.55"),
            ),
        },
        context=validation_label,
    )

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError(f"Invalid {validation_label} profile(s):\n- " + "\n- ".join(invalid))

    jobs: List[Job] = []
    for prof in profiles:
        profile_runtime = dict(runtime_by_profile.get(prof.profile_id) or {})
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_seed_sec = 0.0
            for ds_id in ds_list:
                part_seed_sec += float(profile_runtime.get(ds_id, 180.0))
            part_weight = float(part_seed_sec * len(VALIDATION_SEEDS))
            jobs.append(
                _job(
                    f"{run_family}/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    profile_notes=str(prof.notes),
                    extra_args=list(resolved_common_extra_args + tuple(prof.extra_args)),
                    **common_job_params,
                )
            )
    return jobs


def build_jobs_validation16(
    *,
    dataset_shards: int = 9,
    val15_root: Optional[Path] = None,
) -> List[Job]:
    """Validation-16 focused matrix (11 profiles x 64 datasets x 9 seeds)."""
    return _build_jobs_validation16_family(
        run_family="val16",
        validation_label="validation16",
        validation_datasets=VALIDATION16_DATASETS,
        profile_note_prefix="Val-16",
        dataset_integrity_policy="skip",
        dataset_shards=dataset_shards,
        val15_root=val15_root,
    )


def build_jobs_validation17(
    *,
    dataset_shards: int = 9,
    val15_root: Optional[Path] = None,
) -> List[Job]:
    """Validation-17 focused matrix: Val-16 profile rerun on the packaged after-FS default."""
    return _build_jobs_validation16_family(
        run_family="val17",
        validation_label="validation17",
        validation_datasets=VALIDATION17_DATASETS,
        profile_note_prefix="Val-17",
        dataset_integrity_policy="error",
        common_extra_args=("--df-stage-position", "after_fs"),
        dataset_shards=dataset_shards,
        val15_root=val15_root,
    )


def _balanced_shard_assign_validation16_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-16 by dataset partitions (group all profile bundles per partition)."""
    return _balanced_shard_assign_profile_bundles(
        jobs,
        num_shards,
        run_family="val16",
        profile_ids=VALIDATION16_PROFILE_MANIFEST.keys(),
    )


def _balanced_shard_assign_validation17_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-17 by dataset partitions (group all profile bundles per partition)."""
    return _balanced_shard_assign_profile_bundles(
        jobs,
        num_shards,
        run_family="val17",
        profile_ids=VALIDATION17_PROFILE_MANIFEST.keys(),
    )


def _balanced_shard_assign_profile_bundles(
    jobs: Sequence[Job],
    num_shards: int,
    *,
    run_family: str,
    profile_ids: Iterable[str],
) -> Dict[int, List[str]]:
    """Shard a validation family by dataset partitions (group all profile bundles per partition)."""
    profile_re = re.compile(
        r"^"
        + re.escape(run_family)
        + r"/("
        + "|".join(re.escape(p) for p in sorted(profile_ids))
        + r")/(ds\d+)$"
    )
    grouped: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = profile_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        grouped.setdefault(part_id, {})[profile_id] = job

    bundle_items: List[Tuple[float, List[Job]]] = []
    for _, bundle in grouped.items():
        bundle_weight = sum(float(j.weight) for j in bundle.values())
        bundle_items.append((bundle_weight, list(bundle.values())))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for bundle_weight, bundle_jobs in sorted(bundle_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        for j in sorted(bundle_jobs, key=lambda j: j.job_id):
            shards[target].append(j.job_id)
        totals[target] += float(bundle_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


# ---------------------------------------------------------------------------
# Validation-18 coverage-first campaign (5 plan kinds)
# ---------------------------------------------------------------------------

VAL18_FULL64: List[str] = list(VALIDATION17_DATASETS)

VAL18_DIAG24: List[str] = [
    "colon_alon",
    "leukemia_golub",
    "ovarian_petricoin",
    "srbct_khan",
    "prostate_singh",
    "lymphoma_3",
    "cns_pomeroy",
    "gli_85",
    "arcene_nips03",
    "tox_171",
    "pixraw10p",
    "smk_can_187",
    "cumida_brain_gse50161",
    "cumida_leukemia_subtypes",
    "carcinom_11class",
    "xena_tcga_gbm",
    "dorothea_nips03",
    "xena_tcga_brca",
    "nci60_strict_holdout",
    "nci9_60_9class",
    "gcm_ramaswamy",
    "tumor11_su",
    "gisette_nips03",
    "rv_basehock",
]

# All 39 FS methods for singleton sweep (sorted for reproducibility).
VAL18_SINGLETON_METHODS: Tuple[str, ...] = (
    "anova_f",
    "boruta",
    "chi_square",
    "class_pareto_front",
    "cluster_stability",
    "cmim",
    "copula_knockoff",
    "decorrelated_stability",
    "dove_class_specific",
    "ecoc_class_aware",
    "fcbf",
    "gradient_boosting",
    "group_sparse_lasso",
    "hsic_lasso",
    "ipss",
    "iterative_redundancy_pruning",
    "iterative_redundancy_pruning_bounded",
    "joint_auc_l1",
    "joint_multiclass_support",
    "ktsp",
    "linear_svm",
    "mrmr_jmi",
    "mutual_information",
    "nearest_shrunken_centroid",
    "oaenet",
    "ova_ensemble",
    "pfc_sdr",
    "relieff",
    "rfecv",
    "save_sdr",
    "sir_sdr",
    "slce_centroid_encoder",
    "sparse_multinomial",
    "stability_lasso",
    "stability_subsample",
    "subspace_stability",
    "tigress_stability",
    "treeshap",
    "wmw_auc",
)

VAL18_CLASSIFIER_UNIVERSE: Tuple[str, ...] = (
    "lr",
    "elastic_net_lr",
    "svm_linear",
    "dlda",
    "shrinkage_lda",
    "nsc",
    "pls_da_classifier",
    "nb",
    "svm_rbf",
    "gpc",
    "knn",
    "vote_ensemble",
    "tabpfn",
    "rf",
    "extra_tree",
    "xgb",
    "lgbm",
    "catboost",
    "rp_ensemble",
    "dbda",
    "gqda",
    "bc_svm_linear",
    "sglnn",
    "rff_lr",
    "near_subspace",
    "spatial_median_da",
    "copula_da",
    "tabm",
    "realmlp",
    "cpda",
)

VAL19_ADDED_CLASSIFIERS: Tuple[str, ...] = (
    "rff_lr",
    "near_subspace",
    "spatial_median_da",
    "copula_da",
    "tabm",
    "realmlp",
    "cpda",
)

VAL19_HDLSS_EXTREME_OLD: Tuple[str, ...] = (
    "lr",
    "elastic_net_lr",
    "svm_linear",
    "bc_svm_linear",
    "rp_ensemble",
    "dlda",
    "shrinkage_lda",
    "nsc",
    "pls_da_classifier",
    "nb",
    "dbda",
    "gqda",
    "sglnn",
)

VAL19_HDLSS_EXTREME_NEW: Tuple[str, ...] = VAL19_HDLSS_EXTREME_OLD + VAL19_ADDED_CLASSIFIERS

VAL19_HDLSS_MODERATE_OLD: Tuple[str, ...] = (
    "lr",
    "elastic_net_lr",
    "svm_linear",
    "bc_svm_linear",
    "rp_ensemble",
    "svm_rbf",
    "dlda",
    "shrinkage_lda",
    "nsc",
    "pls_da_classifier",
    "gpc",
    "nb",
    "knn",
    "vote_ensemble",
    "tabpfn",
    "dbda",
    "gqda",
    "sglnn",
)

VAL19_HDLSS_MODERATE_NEW: Tuple[str, ...] = VAL19_HDLSS_MODERATE_OLD + VAL19_ADDED_CLASSIFIERS

# Val-19 launches requested on lab01-lab04 only. TabPFN is not available on all
# four CPU hosts, so the CPU-fleet matched pool surface uses a frozen moderate
# snapshot with TabPFN removed to keep every shard host-stable.
VAL19_HDLSS_MODERATE_CPU_OLD: Tuple[str, ...] = tuple(
    clf for clf in VAL19_HDLSS_MODERATE_OLD if clf != "tabpfn"
)
VAL19_HDLSS_MODERATE_CPU_NEW: Tuple[str, ...] = VAL19_HDLSS_MODERATE_CPU_OLD + VAL19_ADDED_CLASSIFIERS

VAL18_CPU_HOSTS: Tuple[str, ...] = (
    "host0.example.com",
    "host0.example.com",
    "host0.example.com",
    "host0.example.com",
)

VAL18_CPU_HOST_CAPACITY_CORES: Dict[str, int] = {
    "host0.example.com": 64,
    "host0.example.com": 32,
    "host0.example.com": 64,
    "host0.example.com": 64,
}

VAL18_GPU_HOSTS: Tuple[str, ...] = ("arch-ml",)

VAL18_HOST_WORKER_TARGETS: Dict[str, Dict[str, int]] = {
    "host0.example.com": {"pods_per_host": 2, "max_workers_per_pod": 32, "target_total_workers": 64},
    "host0.example.com": {"pods_per_host": 1, "max_workers_per_pod": 32, "target_total_workers": 32},
    "host0.example.com": {"pods_per_host": 2, "max_workers_per_pod": 32, "target_total_workers": 64},
    "host0.example.com": {"pods_per_host": 2, "max_workers_per_pod": 32, "target_total_workers": 64},
    "arch-ml": {"pods_per_host": 2, "max_workers_per_pod": 10, "target_total_workers": 20},
}

# Val-19 is a memory-sensitive CPU-only rerun surface. Keep the recorded host
# targets conservative so the generated work split matches the shard-level RAM
# guidance (`MAX_WORKERS<=4`) and the recent Val-18 OOM envelope.
VAL19_HOST_WORKER_TARGETS: Dict[str, Dict[str, int]] = {
    "host0.example.com": {"pods_per_host": 2, "max_workers_per_pod": 2, "target_total_workers": 4},
    "host0.example.com": {"pods_per_host": 1, "max_workers_per_pod": 4, "target_total_workers": 4},
    "host0.example.com": {"pods_per_host": 2, "max_workers_per_pod": 2, "target_total_workers": 4},
    "host0.example.com": {"pods_per_host": 2, "max_workers_per_pod": 2, "target_total_workers": 4},
}

VAL20_CPU_HOSTS: Tuple[str, ...] = (
    "host0.example.com",
    "host0.example.com",
)

VAL20_GPU_HOSTS: Tuple[str, ...] = ("arch-ml",)
VAL20_TABARENA_HOSTS: Tuple[str, ...] = (*VAL20_CPU_HOSTS, *VAL20_GPU_HOSTS)

VAL20_HOST_CAPACITY_CORES: Dict[str, int] = {
    "host0.example.com": 64,
    "host0.example.com": 32,
    "arch-ml": 24,
}

VAL20_HOST_WORKER_TARGETS: Dict[str, Dict[str, int]] = {
    "host0.example.com": {"pods_per_host": 4, "max_workers_per_pod": 7, "target_total_workers": 28},
    "host0.example.com": {"pods_per_host": 3, "max_workers_per_pod": 5, "target_total_workers": 14},
    "arch-ml": {"pods_per_host": 2, "max_workers_per_pod": 5, "target_total_workers": 10},
}


def _replace_flag_value(flags: Sequence[str], flag: str, value: str) -> Tuple[str, ...]:
    """Return a copy of ``flags`` with the first ``flag`` value replaced."""
    out = list(flags)
    try:
        idx = out.index(str(flag))
    except ValueError:
        out.extend([str(flag), str(value)])
        return tuple(out)
    if idx + 1 >= len(out):
        out.append(str(value))
    else:
        out[idx + 1] = str(value)
    return tuple(out)


def _remove_flag_and_value(flags: Sequence[str], flag: str) -> Tuple[str, ...]:
    """Return ``flags`` with every occurrence of ``flag`` and its value removed."""
    out: List[str] = []
    skip_next = False
    for item in flags:
        if skip_next:
            skip_next = False
            continue
        if str(item) == str(flag):
            skip_next = True
            continue
        out.append(str(item))
    return tuple(out)


def _remove_flag(flags: Sequence[str], flag: str) -> Tuple[str, ...]:
    """Return ``flags`` with all occurrences of a standalone flag removed."""
    return tuple(str(item) for item in flags if str(item) != str(flag))


def _remove_prefixed_flags(flags: Sequence[str], prefix: str) -> Tuple[str, ...]:
    """Remove all flags starting with ``prefix`` plus any attached value token."""
    out: List[str] = []
    skip_next = False
    prefix = str(prefix)
    for idx, item in enumerate(flags):
        if skip_next:
            skip_next = False
            continue
        token = str(item)
        if token.startswith(prefix):
            if idx + 1 < len(flags) and not str(flags[idx + 1]).startswith("--"):
                skip_next = True
            continue
        out.append(token)
    return tuple(out)


def _val18_shared_stage_flags() -> Tuple[str, ...]:
    """Shared scaffold flags for Val-18 profiles (after_fs default)."""
    return (
        "--df-stage-position", "after_fs",
        "--df-family-set", "flex",
        "--df-compute-ad",
        "--df-compute-qq-pp",
        "--df-compute-dip",
        "--df-interval-likelihood",
        "--df-compute-crps",
        "--df-crps-uq-decomposition",
        "--df-lmoment-prescreen",
        "--df-lmoment-prescreen-max-candidates", "12",
        "--folding-method", "pls_da",
        "--enable-prefilter-rnaseq-nb-lrt",
        "--prefilter-rnaseq-nb-lrt-alpha", "0.10",
        "--enable-classifier-conformal",
        "--classifier-conformal-alpha", "0.10",
        "--classifier-conformal-calibration-fraction", "0.25",
        "--classifier-conformal-min-calibration", "20",
        "--enable-stage2-ratio-augmentation",
        "--stage2-ratio-max-features", "16",
        "--stage2-ratio-selection-method", "correlation",
        "--enable-model-cv-runtime-containment",
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
    )


def _val18_a_control_flags() -> Tuple[str, ...]:
    """A01 simple anchor flags."""
    legacy_models = ("lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr")
    return (
        *_val18_shared_stage_flags(),
        "--dist-criterion", "simple",
        "--mnpo-performance-oracle-mode", "single",
        "--fs-oracle-weighting-mode", "uniform",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates", *legacy_models,
        "--model-cv-runtime-max-candidates", "8",
        "--fs-copula-derandomize-runs", "3",
    )


def _val18_d_default_flags() -> Tuple[str, ...]:
    """A02 production anchor flags.

    Val-17 CC-V17-3/CC-V17-4 promotions folded in:
    - CLP (--fs-fold-preference-mode logistic)
    - Payoff shrinkage (--fs-payoff-shrinkage-kappa 0.15)
    - Conformal efficiency (--fs-use-conformal-efficiency, APS)
    - JS oracle-weight shrinkage (--fs-oracle-weight-js-shrinkage)
    """
    legacy_models = ("lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr")
    return (
        *_val18_shared_stage_flags(),
        "--dist-criterion", "simple",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--prefilter-bh-ttest-enabled",
        "--prefilter-variance-floor-enabled",
        "--fs-copula-derandomize-runs", "3",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
        # Val-17 full-stack promotions (CC-V17-3 p=0.033, CC-V17-4 CLP trending).
        "--fs-fold-preference-mode", "logistic",
        "--fs-payoff-shrinkage-kappa", "0.15",
        "--fs-use-conformal-efficiency",
        "--fs-conformal-efficiency-method", "aps",
        "--fs-oracle-weight-js-shrinkage",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates", *legacy_models,
        "--model-cv-runtime-max-candidates", "8",
    )


def _val18_ref_flags() -> Tuple[str, ...]:
    """A03 corrected reference anchor flags (v16_ref lineage)."""
    return (
        *_val18_d_default_flags(),
        "--regime-gating-enabled",
        "--regime-gating-difficulty-source", "historical",
        "--regime-gating-target-tier", "very_hard",
        "--regime-gating-min-samples-per-class", "7",
        "--regime-gating-very-hard-min-classes", "5",
        "--regime-gating-low-p-over-n-threshold", "0",
        "--regime-gating-simple-fs-method-set", "strict_plus_mrmr",
        "--regime-gating-very-hard-portfolio-max-methods", "4",
        "--regime-gating-very-hard-copula-derandomize-runs", "5",
        "--regime-gating-extreme-multiclass-enabled",
        "--regime-gating-extreme-multiclass-threshold", "8",
        "--regime-gating-extreme-multiclass-min-samples-per-class", "11",
        "--fs-copula-derandomize-runs", "5",
        "--fs-max-selected-features-ratio", "0.5",
        "--fs-max-selected-features-cap", "500",
        "--fs-stability-threshold-method", "fixed",
    )


def _val18_common_job_params(*, seeds: Sequence[int]) -> Dict[str, Any]:
    """Shared Job params for Val-18 plan kinds."""
    return dict(
        seeds=list(seeds),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="error",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        progress_stall_watchdog_sec=1800,
        task_timeout_sec=21600,
        fs_method_timeout_sec=3600,
    )


def _val18_runtime_estimate(
    dataset_ids: Sequence[str],
    benchmark_datasets: Mapping[str, Any],
    val17_runtime: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute per-dataset runtime estimates from Val-17 history (or fallback)."""
    est: Dict[str, float] = {}
    ref_hist = dict(val17_runtime.get("v16_ref") or {})
    a_hist = dict(val17_runtime.get("a_control") or {})
    for ds_id in dataset_ids:
        spec = benchmark_datasets.get(ds_id)
        t = float(ref_hist.get(ds_id, 0.0))
        if t <= 0.0:
            t = float(a_hist.get(ds_id, 0.0))
        if t <= 0.0:
            t = float(max(60.0, _dataset_weight(spec) * 250.0))
        est[ds_id] = float(t)
    return est


def _val18_validate_method_sets(profiles: Sequence[BenchmarkProfile], context: str) -> None:
    """Validate all profiles reference valid, non-deprecated fs_method_sets."""
    fs_method_sets = _load_fs_method_sets()
    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError(f"Invalid {context} profile(s):\n- " + "\n- ".join(invalid))


def _job_execution_lane(job: Job) -> str:
    """Infer the execution lane for a generated job."""
    params = dict(job.params or {})
    lane = str(params.get("execution_lane", "") or "").strip().lower()
    if lane:
        return lane
    return "tabpfn" if "tabpfn" in str(job.job_id).lower() else "cpu"


def _job_preferred_hosts(job: Job) -> Tuple[str, ...]:
    """Return preferred hosts for a generated job, filling Val-18 defaults when omitted."""
    params = dict(job.params or {})
    hosts = tuple(str(host).strip() for host in list(params.get("preferred_hosts") or []) if str(host).strip())
    if hosts:
        return hosts
    return VAL18_GPU_HOSTS if _job_execution_lane(job) == "tabpfn" else VAL18_CPU_HOSTS


def _recommended_host_assignment(
    shards: Dict[int, List[str]],
    jobs_by_id: Dict[str, Job],
    *,
    cpu_hosts: Sequence[str],
    gpu_hosts: Sequence[str],
    cpu_host_capacity_cores: Dict[str, int],
    host_worker_targets: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    """Recommend a host mapping that balances CPU shard weight by host capacity."""
    shard_totals = _shard_weight_totals(shards, jobs_by_id)
    host_to_shards: Dict[str, List[int]] = {
        host: [] for host in (*cpu_hosts, *gpu_hosts)
    }
    host_assigned_weight: Dict[str, float] = {host: 0.0 for host in host_to_shards}
    shard_to_host: Dict[int, str] = {}

    cpu_shards: List[Tuple[float, int]] = []
    tabpfn_shards: List[Tuple[float, int]] = []
    for shard_id, job_ids in shards.items():
        if not job_ids:
            continue
        shard_jobs = [jobs_by_id[jid] for jid in job_ids if jid in jobs_by_id]
        lanes = {_job_execution_lane(job) for job in shard_jobs}
        weight = float(shard_totals.get(int(shard_id), 0.0))
        if lanes == {"tabpfn"}:
            tabpfn_shards.append((weight, int(shard_id)))
        else:
            cpu_shards.append((weight, int(shard_id)))

    for weight, shard_id in sorted(cpu_shards, key=lambda x: float(x[0]), reverse=True):
        target = min(
            cpu_hosts,
            key=lambda host: (
                float(host_assigned_weight[host]) / float(cpu_host_capacity_cores[host]),
                float(host_assigned_weight[host]),
                host,
            ),
        )
        host_to_shards[target].append(int(shard_id))
        host_assigned_weight[target] += float(weight)
        shard_to_host[int(shard_id)] = str(target)

    for weight, shard_id in sorted(tabpfn_shards, key=lambda x: float(x[0]), reverse=True):
        if not gpu_hosts:
            continue
        target = gpu_hosts[0]
        host_to_shards[target].append(int(shard_id))
        host_assigned_weight[target] += float(weight)
        shard_to_host[int(shard_id)] = str(target)

    host_summary: Dict[str, Dict[str, Any]] = {}
    for host in (*cpu_hosts, *gpu_hosts):
        assigned_weight = float(host_assigned_weight[host])
        capacity_cores = int(cpu_host_capacity_cores.get(host, 0))
        normalized = assigned_weight / float(capacity_cores) if capacity_cores > 0 else assigned_weight
        host_summary[host] = {
            "shards": list(host_to_shards[host]),
            "assigned_weight": assigned_weight,
            "capacity_cores": capacity_cores,
            "normalized_weight_per_core": normalized,
            "worker_target": dict(host_worker_targets.get(host, {})),
        }

    return {
        "shard_to_host": {str(k): v for k, v in shard_to_host.items()},
        "host_summary": host_summary,
    }


def _validation18_recommended_host_assignment(
    shards: Dict[int, List[str]],
    jobs_by_id: Dict[str, Job],
) -> Dict[str, Any]:
    """Recommend a Val-18 host mapping that balances CPU shard weight by host capacity."""
    return _recommended_host_assignment(
        shards,
        jobs_by_id,
        cpu_hosts=VAL18_CPU_HOSTS,
        gpu_hosts=VAL18_GPU_HOSTS,
        cpu_host_capacity_cores=VAL18_CPU_HOST_CAPACITY_CORES,
        host_worker_targets=VAL18_HOST_WORKER_TARGETS,
    )


def _validation19_recommended_host_assignment(
    shards: Dict[int, List[str]],
    jobs_by_id: Dict[str, Job],
) -> Dict[str, Any]:
    """Recommend a Val-19 CPU-only host mapping with conservative worker caps."""
    return _recommended_host_assignment(
        shards,
        jobs_by_id,
        cpu_hosts=VAL18_CPU_HOSTS,
        gpu_hosts=(),
        cpu_host_capacity_cores=VAL18_CPU_HOST_CAPACITY_CORES,
        host_worker_targets=VAL19_HOST_WORKER_TARGETS,
    )


def _validation20_recommended_host_assignment(
    shards: Dict[int, List[str]],
    jobs_by_id: Dict[str, Job],
) -> Dict[str, Any]:
    """Recommend a Val-20 host mapping for the 3-host mixed CPU/GPU setup."""
    return _recommended_host_assignment(
        shards,
        jobs_by_id,
        cpu_hosts=VAL20_CPU_HOSTS,
        gpu_hosts=VAL20_GPU_HOSTS,
        cpu_host_capacity_cores=VAL20_HOST_CAPACITY_CORES,
        host_worker_targets=VAL20_HOST_WORKER_TARGETS,
    )


def _validation20_tabarena_recommended_host_assignment(
    shards: Dict[int, List[str]],
    jobs_by_id: Dict[str, Job],
) -> Dict[str, Any]:
    """Recommend a Val-20 TabArena host mapping across the 3-host CPU companion lane."""
    return _recommended_host_assignment(
        shards,
        jobs_by_id,
        cpu_hosts=VAL20_TABARENA_HOSTS,
        gpu_hosts=(),
        cpu_host_capacity_cores=VAL20_HOST_CAPACITY_CORES,
        host_worker_targets=VAL20_HOST_WORKER_TARGETS,
    )


def _val18_build_jobs(
    *,
    run_family: str,
    profiles: Sequence[BenchmarkProfile],
    dataset_ids: Sequence[str],
    seeds: Sequence[int],
    dataset_shards: int,
    runtime_est: Dict[str, float],
    extra_common: Sequence[str] = (),
) -> List[Job]:
    """Generic Val-18 job builder common to all plan kinds."""
    hf_meta = _load_hf_manifest_metadata()
    hf_ids = set(hf_meta.keys())
    if not hf_ids:
        raise RuntimeError(f"HF bundle manifest metadata is empty; cannot build {run_family} plan.")

    validation_ids = [str(ds) for ds in dataset_ids if str(ds) in hf_ids]
    if not validation_ids:
        raise RuntimeError(f"No datasets available in HF bundle manifests for {run_family}.")

    missing_hf = [str(ds) for ds in dataset_ids if str(ds) not in hf_ids]
    if missing_hf:
        preview = ", ".join(missing_hf[:10])
        suffix = " ..." if len(missing_hf) > 10 else ""
        print(
            f"[{run_family}] Skipping {len(missing_hf)} dataset(s) not in HF manifests: "
            f"{preview}{suffix}",
            file=sys.stderr,
        )

    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        per_seed = float(runtime_est.get(ds_id, 180.0))
        ds_items.append((str(ds_id), per_seed * float(len(seeds)) * float(len(profiles))))
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_extra = ("--emit-summary", "--compute-budget", "standard", *tuple(extra_common))
    common_params = _val18_common_job_params(seeds=seeds)

    jobs: List[Job] = []
    for prof in profiles:
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_weight = sum(float(runtime_est.get(d, 180.0)) for d in ds_list) * float(len(seeds))
            part_weight *= float(prof.weight_mult)
            profile_job_params = dict(prof.job_params or {})
            jobs.append(
                _job(
                    f"{run_family}/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    profile_notes=str(prof.notes),
                    extra_args=list(common_extra + tuple(prof.extra_args)),
                    **profile_job_params,
                    **common_params,
                )
            )
    return jobs


TABARENA_DEFAULT_OFFICIAL_FOLDS = 10


def _tabarena_dataset_weight(spec: Any) -> float:
    """Rough runtime weight for a TabArena dataset/task."""
    n_samples = float(max(1, int(getattr(spec, "n_samples", 1) or 1)))
    n_features = float(max(1, int(getattr(spec, "n_features", 1) or 1)))
    n_classes = float(max(2, int(getattr(spec, "n_classes", 2) or 2)))
    problem_type = str(getattr(spec, "problem_type", "binary") or "binary").strip().lower()

    base = 0.8
    base += 0.9 * math.log10(n_samples + 10.0)
    base += 0.45 * math.log10(n_features + 10.0)
    if problem_type == "multiclass":
        base += 0.35 + min(0.30, 0.05 * max(0.0, n_classes - 2.0))
    return float(max(1.0, base))


def _tabarena_fold_multiplier(protocol: str, official_fold_limit: int) -> int:
    proto = str(protocol or "openml_task").strip().lower()
    if proto != "openml_task":
        return 1
    limit = int(official_fold_limit or 0)
    return int(limit if limit > 0 else TABARENA_DEFAULT_OFFICIAL_FOLDS)


def _resolve_tabarena_dataset_ids(
    *,
    dataset_sets: Sequence[str],
    datasets: Sequence[str],
    exclude_datasets: Sequence[str],
    dataset_specs: Mapping[str, Any],
    dataset_set_registry: Mapping[str, Sequence[str]],
) -> List[str]:
    selected: List[str] = []
    for set_name in dataset_sets:
        key = str(set_name)
        if key not in dataset_set_registry:
            raise KeyError(f"Unknown TabArena dataset set: {key}")
        selected.extend(str(ds_id) for ds_id in dataset_set_registry[key])
    selected.extend(str(ds_id) for ds_id in datasets)

    if not selected:
        raise ValueError("TabArena plan profile selected no datasets.")

    exclude = {str(ds_id) for ds_id in exclude_datasets}
    out: List[str] = []
    seen: set[str] = set()
    for ds_id in selected:
        if ds_id not in dataset_specs:
            raise KeyError(f"Unknown TabArena dataset: {ds_id}")
        if ds_id in exclude or ds_id in seen:
            continue
        seen.add(ds_id)
        out.append(ds_id)

    if not out:
        raise ValueError("TabArena plan profile selected no datasets after exclusions.")
    return out


def _build_tabarena_jobs(
    *,
    run_family: str,
    profiles: Sequence[TabArenaPlanProfile],
    dataset_shards: int,
) -> List[Job]:
    """Build sharded TabArena benchmark jobs."""
    dataset_specs, dataset_set_registry = _load_tabarena_registry()
    task_shards = int(max(1, dataset_shards))

    jobs: List[Job] = []
    for prof in profiles:
        resolved_dataset_ids = _resolve_tabarena_dataset_ids(
            dataset_sets=prof.dataset_sets,
            datasets=prof.datasets,
            exclude_datasets=prof.exclude_datasets,
            dataset_specs=dataset_specs,
            dataset_set_registry=dataset_set_registry,
        )
        fold_mult = _tabarena_fold_multiplier(prof.protocol, prof.official_fold_limit)
        total_weight = sum(_tabarena_dataset_weight(dataset_specs[ds_id]) for ds_id in resolved_dataset_ids)
        total_weight *= float(max(1, len(tuple(prof.seeds))))
        total_weight *= float(max(1, fold_mult))
        total_weight *= float(prof.weight_mult)
        shard_weight = float(total_weight) / float(task_shards)

        profile_job_params = dict(prof.job_params or {})
        for task_shard_index in range(task_shards):
            jobs.append(
                _job(
                    f"{run_family}/{prof.profile_id}/ts{task_shard_index + 1:02d}",
                    "tabarena_benchmark",
                    weight=shard_weight,
                    dataset_sets=list(prof.dataset_sets),
                    datasets=list(prof.datasets),
                    exclude_datasets=list(prof.exclude_datasets),
                    resolved_dataset_ids=list(resolved_dataset_ids),
                    profile=str(prof.profile),
                    protocol=str(prof.protocol),
                    seeds=[int(seed) for seed in tuple(prof.seeds)],
                    official_fold_limit=int(prof.official_fold_limit),
                    task_shard_count=int(task_shards),
                    task_shard_index=int(task_shard_index),
                    estimated_task_count_total=int(len(resolved_dataset_ids) * len(tuple(prof.seeds)) * fold_mult),
                    estimated_task_count_shard=int(
                        math.ceil((len(resolved_dataset_ids) * len(tuple(prof.seeds)) * fold_mult) / float(task_shards))
                    ),
                    profile_notes=str(prof.notes),
                    quiet=bool(prof.quiet),
                    skip_official_leaderboard=bool(prof.skip_official_leaderboard),
                    leaderboard_method_name=str(prof.leaderboard_method_name),
                    extra_args=list(prof.extra_args),
                    **profile_job_params,
                )
            )
    return jobs


# ---- Family A: Anchors And Bypass Controls ----

VALIDATION18_ANCHORS_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "A01_simple_anchor_after_fs": {"anchor": "yes", "contrast_ref": "A03_ref_anchor_after_fs", "effect": "simple_anchor"},
    "A02_default_anchor_after_fs": {"anchor": "yes", "contrast_ref": "A03_ref_anchor_after_fs", "effect": "production_anchor"},
    "A03_ref_anchor_after_fs": {"anchor": "yes", "contrast_ref": "A01_simple_anchor_after_fs", "effect": "corrected_reference"},
    "A04_skip_fs_all_features": {"anchor": "no", "contrast_ref": "A03_ref_anchor_after_fs", "effect": "skip_fs_control"},
    "A05_skip_df_ref": {"anchor": "no", "contrast_ref": "A03_ref_anchor_after_fs", "effect": "skip_df_control"},
    "A06_skip_fs_skip_df": {"anchor": "no", "contrast_ref": "A03_ref_anchor_after_fs", "effect": "skip_fs_skip_df_control"},
    "A07_ref_before_fs": {"anchor": "no", "contrast_ref": "A03_ref_anchor_after_fs", "effect": "ordering_control"},
    "A08_ref_no_regime_gating": {"anchor": "no", "contrast_ref": "A03_ref_anchor_after_fs", "effect": "gating_control"},
}


def build_jobs_validation18_anchors(
    *,
    dataset_shards: int = 9,
    val17_root: Optional[Path] = None,
) -> List[Job]:
    """Val-18 Family A: 8 anchor/control profiles x 64 datasets x 9 seeds."""
    benchmark_datasets, _ = _load_benchmark_registry()
    if val17_root is None:
        val17_root = REPO_ROOT / "run_artifacts" / "validation-17"
    val17_runtime = _load_runtime_hints_from_summaries(val17_root, phase_tag="val17")
    validation_ids = list(VAL18_FULL64)
    runtime_est = _val18_runtime_estimate(validation_ids, benchmark_datasets, val17_runtime)

    # Skip-FS flags: force all-features bypass via huge threshold.
    skip_fs_flags: Tuple[str, ...] = (
        *_val18_shared_stage_flags(),
        "--dist-criterion", "simple",
        "--fs-max-selected-features-ratio", "1.0",
        "--fs-max-selected-features-cap", "999999",
        "--regime-gating-low-p-over-n-threshold", "999999",
        "--mnpo-performance-oracle-mode", "single",
        "--fs-oracle-weighting-mode", "uniform",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates", "dlda",
    )

    # Skip-DF flags: ref + disable CDF transform.
    skip_df_flags: Tuple[str, ...] = (*_val18_ref_flags(), "--disable-cdf-transform")

    # Skip-FS + skip-DF combined.
    skip_fs_skip_df_flags: Tuple[str, ...] = (*skip_fs_flags, "--disable-cdf-transform")

    # Before-FS ordering control.
    ref_before_fs_flags: Tuple[str, ...] = tuple(
        ("--df-stage-position", "before_fs") if i == 0 else (v,)
        for i, v in enumerate(_val18_ref_flags())
        for _ in [None]  # flatten
    )
    # Rebuild: replace after_fs with before_fs in ref flags.
    ref_flags_list = list(_val18_ref_flags())
    try:
        idx = ref_flags_list.index("after_fs")
        ref_flags_list[idx] = "before_fs"
    except ValueError:
        pass
    ref_before_fs_flags = tuple(ref_flags_list)

    # No regime gating control.
    ref_no_gating_flags = _remove_prefixed_flags(_val18_ref_flags(), "--regime-gating")

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile("A01_simple_anchor_after_fs", "strict_plus_mrmr", _val18_a_control_flags(),
                         notes="Val-18 simple anchor: a_control-style stack, after_fs, simple DF, legacy clf."),
        BenchmarkProfile("A02_default_anchor_after_fs", "mnpo_broad_all", _val18_d_default_flags(),
                         notes="Val-18 production anchor: current d_default on packaged after_fs."),
        BenchmarkProfile("A03_ref_anchor_after_fs", "mnpo_v14_core_plus_ipss", _val18_ref_flags(),
                         notes="Val-18 corrected reference anchor: v16_ref lineage on after_fs."),
        BenchmarkProfile("A04_skip_fs_all_features", "strict_plus_mrmr", skip_fs_flags,
                         notes="Val-18 skip-FS control: all-features bypass, DF kept on."),
        BenchmarkProfile("A05_skip_df_ref", "mnpo_v14_core_plus_ipss", skip_df_flags,
                         notes="Val-18 skip-DF control: A03 + disable-cdf-transform."),
        BenchmarkProfile("A06_skip_fs_skip_df", "strict_plus_mrmr", skip_fs_skip_df_flags,
                         notes="Val-18 pure no-FS/no-DF control."),
        BenchmarkProfile("A07_ref_before_fs", "mnpo_v14_core_plus_ipss", ref_before_fs_flags,
                         notes="Val-18 ordering control: A03 but df_stage_position=before_fs."),
        BenchmarkProfile("A08_ref_no_regime_gating", "mnpo_v14_core_plus_ipss", ref_no_gating_flags,
                         notes="Val-18 gating control: A03 but regime gating disabled."),
    ]

    _val18_validate_method_sets(profiles, "validation18_anchors")

    return _val18_build_jobs(
        run_family="val18_anchors",
        profiles=profiles,
        dataset_ids=validation_ids,
        seeds=list(VALIDATION_SEEDS),
        dataset_shards=dataset_shards,
        runtime_est=runtime_est,
    )


def _balanced_shard_assign_validation18_anchors_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    return _balanced_shard_assign_profile_bundles(
        jobs, num_shards,
        run_family="val18_anchors",
        profile_ids=VALIDATION18_ANCHORS_PROFILE_MANIFEST.keys(),
    )


# ---- Family M: Full Singleton Feature-Selection Sweep ----

VALIDATION18_SINGLETONS_PROFILE_IDS: List[str] = []
for _m in VAL18_SINGLETON_METHODS:
    VALIDATION18_SINGLETONS_PROFILE_IDS.append(f"M_RAW_{_m}")
    VALIDATION18_SINGLETONS_PROFILE_IDS.append(f"M_SCAFFOLD_{_m}")
del _m


def build_jobs_validation18_singletons(
    *,
    dataset_shards: int = 9,
    val17_root: Optional[Path] = None,
) -> List[Job]:
    """Val-18 Family M: 39 methods x 2 variants (raw/scaffold) x 64 datasets x 5 seeds."""
    benchmark_datasets, _ = _load_benchmark_registry()
    if val17_root is None:
        val17_root = REPO_ROOT / "run_artifacts" / "validation-17"
    val17_runtime = _load_runtime_hints_from_summaries(val17_root, phase_tag="val17")
    validation_ids = list(VAL18_FULL64)
    runtime_est = _val18_runtime_estimate(validation_ids, benchmark_datasets, val17_runtime)

    seeds = [11, 23, 37, 42, 59]  # 5 seeds for initial pass

    # M-RAW flags: DF off, no gating, no screening, no folding, no BH/VF, fixed dlda.
    raw_base_flags: Tuple[str, ...] = (
        "--df-stage-position", "after_fs",
        "--disable-cdf-transform",
        "--dist-criterion", "simple",
        "--mnpo-performance-oracle-mode", "single",
        "--fs-oracle-weighting-mode", "uniform",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates", "dlda",
        "--model-cv-runtime-max-candidates", "1",
        "--fs-copula-derandomize-runs", "3",
    )

    # M-SCAFFOLD flags: after_fs, simple DF, BH/VF on, folding pls_da, no regime gating, fixed dlda.
    scaffold_base_flags: Tuple[str, ...] = (
        "--df-stage-position", "after_fs",
        "--df-family-set", "flex",
        "--df-compute-ad",
        "--df-compute-qq-pp",
        "--df-compute-dip",
        "--df-interval-likelihood",
        "--df-compute-crps",
        "--df-crps-uq-decomposition",
        "--df-lmoment-prescreen",
        "--df-lmoment-prescreen-max-candidates", "12",
        "--dist-criterion", "simple",
        "--folding-method", "pls_da",
        "--prefilter-bh-ttest-enabled",
        "--prefilter-variance-floor-enabled",
        "--enable-prefilter-rnaseq-nb-lrt",
        "--prefilter-rnaseq-nb-lrt-alpha", "0.10",
        "--mnpo-performance-oracle-mode", "single",
        "--fs-oracle-weighting-mode", "uniform",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates", "dlda",
        "--model-cv-runtime-max-candidates", "1",
        "--fs-copula-derandomize-runs", "3",
        "--enable-stage2-ratio-augmentation",
        "--stage2-ratio-max-features", "16",
        "--stage2-ratio-selection-method", "correlation",
        "--enable-model-cv-runtime-containment",
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
    )

    profiles: List[BenchmarkProfile] = []
    for method in VAL18_SINGLETON_METHODS:
        fs_set_key = f"singleton_{method}"
        # Raw profile: pure method quality without scaffold help.
        profiles.append(BenchmarkProfile(
            f"M_RAW_{method}", fs_set_key, raw_base_flags,
            notes=f"Val-18 M-RAW singleton: {method} (no scaffold, no DF, fixed dlda).",
        ))
        # Scaffold profile: method compatibility with current scaffold.
        profiles.append(BenchmarkProfile(
            f"M_SCAFFOLD_{method}", fs_set_key, scaffold_base_flags,
            notes=f"Val-18 M-SCAFFOLD singleton: {method} (after_fs scaffold, fixed dlda).",
        ))

    _val18_validate_method_sets(profiles, "validation18_singletons")

    return _val18_build_jobs(
        run_family="val18_singletons",
        profiles=profiles,
        dataset_ids=validation_ids,
        seeds=seeds,
        dataset_shards=dataset_shards,
        runtime_est=runtime_est,
    )


def _balanced_shard_assign_validation18_singletons_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    return _balanced_shard_assign_profile_bundles(
        jobs, num_shards,
        run_family="val18_singletons",
        profile_ids=VALIDATION18_SINGLETONS_PROFILE_IDS,
    )


# ---- Family N: MNPO / Oracle / Stabilizer Matrix ----

VALIDATION18_MNPO_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "N01_core_legacy_voting": {"anchor": "yes", "contrast_ref": "N04_core_banzhaf", "effect": "legacy_vs_mnpo"},
    "N02_core_uniform": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "weighting_uniform"},
    "N03_core_tritrust": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "weighting_tritrust"},
    "N04_core_banzhaf": {"anchor": "yes", "contrast_ref": "N01_core_legacy_voting", "effect": "weighting_banzhaf"},
    "N05_core_shapley": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "weighting_shapley"},
    "N06_core_banzhaf_no_clp": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "ablation_clp"},
    "N07_core_banzhaf_no_payoff": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "ablation_payoff_shrinkage"},
    "N08_core_banzhaf_no_clp_payoff": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "ablation_clp_plus_payoff"},
    "N09_core_banzhaf_no_js": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "ablation_js_shrinkage"},
    "N10_core_banzhaf_no_conformal_eff": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "ablation_conformal_efficiency"},
    "N11_core_banzhaf_cvar": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "cvar_oracle"},
    "N12_core_banzhaf_interaction": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "interaction_oracle"},
    "N13_core_banzhaf_ubayfs": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "ubayfs_oracle"},
    "N14_core_oracle_slim": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "oracle_slim"},
    "N15_core_meta_logistic": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "meta_logistic"},
    # N16_core_meta_dt removed: meta-DT killed (CC-V17-2).
    "N30_core_banzhaf_perf_only": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "oracle_pruning_perf_only"},
    "N31_core_banzhaf_perf_complexity": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "oracle_pruning_perf_complexity"},
    "N32_core_banzhaf_perf_complex_stab": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "oracle_pruning_perf_complexity_stability"},
    "N40_core_banzhaf_5x5": {"anchor": "no", "contrast_ref": "N04_core_banzhaf", "effect": "fold_resolution_5x5"},
    "N41_core_banzhaf_perf_only_5x5": {"anchor": "no", "contrast_ref": "N30_core_banzhaf_perf_only", "effect": "fold_resolution_perf_only_5x5"},
    "N42_core_banzhaf_perf_complex_5x5": {"anchor": "no", "contrast_ref": "N31_core_banzhaf_perf_complexity", "effect": "fold_resolution_perf_complexity_5x5"},
    "N21_broad_legacy_voting": {"anchor": "no", "contrast_ref": "N22_broad_banzhaf", "effect": "broad_legacy_vs_mnpo"},
    "N22_broad_banzhaf": {"anchor": "no", "contrast_ref": "N21_broad_legacy_voting", "effect": "broad_banzhaf"},
    # N23_broad_banzhaf_clp_payoff removed: CLP+payoff now in base ref (CC-V17-3/V17-4).
}


def build_jobs_validation18_mnpo(
    *,
    dataset_shards: int = 9,
    val17_root: Optional[Path] = None,
) -> List[Job]:
    """Val-18 Family N: 25 MNPO/oracle profiles x 64 datasets x 5 seeds."""
    benchmark_datasets, _ = _load_benchmark_registry()
    if val17_root is None:
        val17_root = REPO_ROOT / "run_artifacts" / "validation-17"
    val17_runtime = _load_runtime_hints_from_summaries(val17_root, phase_tag="val17")
    validation_ids = list(VAL18_FULL64)
    runtime_est = _val18_runtime_estimate(validation_ids, benchmark_datasets, val17_runtime)

    seeds = [11, 23, 37, 42, 59]  # 5 seeds

    ref = _val18_ref_flags()

    # N01: legacy voting on core set.
    n01_flags = (*ref, "--selection-strategy", "legacy_voting")
    # N02: uniform weighting.
    n02_flags = _replace_flag_value(ref, "--fs-oracle-weighting-mode", "uniform")
    # N03: tritrust weighting.
    n03_flags = _replace_flag_value(ref, "--fs-oracle-weighting-mode", "tritrust")
    # N04: banzhaf (= ref as-is).
    n04_flags = ref
    # N05: shapley.
    n05_flags = _replace_flag_value(ref, "--fs-oracle-weighting-mode", "shapley")
    # N06-N10: ablations against the promoted full-stack reference.
    n06_flags = _replace_flag_value(ref, "--fs-fold-preference-mode", "vote")
    n07_flags = _replace_flag_value(ref, "--fs-payoff-shrinkage-kappa", "0.0")
    n08_flags = _replace_flag_value(n06_flags, "--fs-payoff-shrinkage-kappa", "0.0")
    n09_flags = _remove_flag(ref, "--fs-oracle-weight-js-shrinkage")
    n10_flags = _remove_flag_and_value(
        _remove_flag(ref, "--fs-use-conformal-efficiency"),
        "--fs-conformal-efficiency-method",
    )
    # N11: banzhaf + CVaR.
    n11_flags = (*ref, "--fs-use-cvar-oracle")
    # N12: banzhaf + interaction oracle.
    n12_flags = (*ref, "--fs-use-interaction-oracle")
    # N13: banzhaf + UBayFS.
    n13_flags = (*ref, "--fs-use-ubayfs-oracle")
    # N14: oracle-slim (single perf oracle + redundancy penalty).
    n14_flags = _replace_flag_value(ref, "--mnpo-performance-oracle-mode", "single")
    # N15: meta-learning logistic.
    n15_flags = (*ref, "--meta-learning-selector", "logistic", "--meta-learning-confidence-threshold", "0.55")
    # N30-N32: oracle-pruning sweeps from VAL18_PLAN.
    n30_flags = (
        *_remove_flag(ref, "--enable-diversity-oracle"),
        "--disable-fs-stability-oracle",
        "--disable-fs-complexity-oracle",
        "--disable-fs-robust-oracle",
    )
    n31_flags = (
        *_remove_flag(ref, "--enable-diversity-oracle"),
        "--disable-fs-stability-oracle",
        "--disable-fs-robust-oracle",
    )
    n32_flags = (
        *_remove_flag(ref, "--enable-diversity-oracle"),
        "--disable-fs-robust-oracle",
    )
    # N40-N42: 5x5 fold-resolution sweep.
    n40_flags = (*ref, "--fs-inner-cv-splits", "5", "--fs-inner-cv-repeats", "5")
    n41_flags = (*n30_flags, "--fs-inner-cv-splits", "5", "--fs-inner-cv-repeats", "5")
    n42_flags = (*n31_flags, "--fs-inner-cv-splits", "5", "--fs-inner-cv-repeats", "5")
    # N16_core_meta_dt removed: meta-DT killed (CC-V17-2).

    # Broad-set profiles (N21-N22).
    # N21: broad + legacy voting.
    n21_flags = (*ref, "--selection-strategy", "legacy_voting")
    # N22: broad + banzhaf.
    n22_flags = ref
    # N23_broad_banzhaf_clp_payoff removed: CLP+payoff now in base ref (CC-V17-3/V17-4).

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile("N01_core_legacy_voting", "mnpo_v14_core_plus_ipss", n01_flags,
                         notes="Val-18 N: core set, legacy_voting selection strategy."),
        BenchmarkProfile("N02_core_uniform", "mnpo_v14_core_plus_ipss", n02_flags,
                         notes="Val-18 N: core set, mnpo_portfolio, uniform weighting."),
        BenchmarkProfile("N03_core_tritrust", "mnpo_v14_core_plus_ipss", n03_flags,
                         notes="Val-18 N: core set, tritrust weighting."),
        BenchmarkProfile("N04_core_banzhaf", "mnpo_v14_core_plus_ipss", n04_flags,
                         notes="Val-18 N: core set, banzhaf weighting (reference)."),
        BenchmarkProfile("N05_core_shapley", "mnpo_v14_core_plus_ipss", n05_flags,
                         notes="Val-18 N: core set, shapley weighting."),
        BenchmarkProfile("N06_core_banzhaf_no_clp", "mnpo_v14_core_plus_ipss", n06_flags,
                         notes="Val-18 N: ablation — remove CLP from full-stack default."),
        BenchmarkProfile("N07_core_banzhaf_no_payoff", "mnpo_v14_core_plus_ipss", n07_flags,
                         notes="Val-18 N: ablation — remove payoff shrinkage from full-stack default."),
        BenchmarkProfile("N08_core_banzhaf_no_clp_payoff", "mnpo_v14_core_plus_ipss", n08_flags,
                         notes="Val-18 N: ablation — remove CLP + payoff shrinkage from full-stack default."),
        BenchmarkProfile("N09_core_banzhaf_no_js", "mnpo_v14_core_plus_ipss", n09_flags,
                         notes="Val-18 N: ablation — remove JS oracle-weight shrinkage from full-stack default."),
        BenchmarkProfile("N10_core_banzhaf_no_conformal_eff", "mnpo_v14_core_plus_ipss", n10_flags,
                         notes="Val-18 N: ablation — remove conformal-efficiency oracle from full-stack default."),
        BenchmarkProfile("N11_core_banzhaf_cvar", "mnpo_v14_core_plus_ipss", n11_flags,
                         notes="Val-18 N: banzhaf + CVaR oracle."),
        BenchmarkProfile("N12_core_banzhaf_interaction", "mnpo_v14_core_plus_ipss", n12_flags,
                         notes="Val-18 N: banzhaf + interaction oracle."),
        BenchmarkProfile("N13_core_banzhaf_ubayfs", "mnpo_v14_core_plus_ipss", n13_flags,
                         notes="Val-18 N: banzhaf + UBayFS oracle."),
        BenchmarkProfile("N14_core_oracle_slim", "mnpo_v14_core_plus_ipss", n14_flags,
                         notes="Val-18 N: single perf oracle + redundancy penalty."),
        BenchmarkProfile("N15_core_meta_logistic", "mnpo_v14_core_plus_ipss", n15_flags,
                         notes="Val-18 N: meta-learning logistic selector."),
        # N16_core_meta_dt removed: meta-DT killed (CC-V17-2).
        BenchmarkProfile("N30_core_banzhaf_perf_only", "mnpo_v14_core_plus_ipss", n30_flags,
                         notes="Val-18 N: oracle-pruning sweep, performance oracle only."),
        BenchmarkProfile("N31_core_banzhaf_perf_complexity", "mnpo_v14_core_plus_ipss", n31_flags,
                         notes="Val-18 N: oracle-pruning sweep, performance + complexity oracles."),
        BenchmarkProfile("N32_core_banzhaf_perf_complex_stab", "mnpo_v14_core_plus_ipss", n32_flags,
                         notes="Val-18 N: oracle-pruning sweep, performance + complexity + stability oracles."),
        BenchmarkProfile("N40_core_banzhaf_5x5", "mnpo_v14_core_plus_ipss", n40_flags,
                         notes="Val-18 N: fold-resolution sweep, default oracle stack with 5x5 inner CV."),
        BenchmarkProfile("N41_core_banzhaf_perf_only_5x5", "mnpo_v14_core_plus_ipss", n41_flags,
                         notes="Val-18 N: fold-resolution sweep, perf-only oracle stack with 5x5 inner CV."),
        BenchmarkProfile("N42_core_banzhaf_perf_complex_5x5", "mnpo_v14_core_plus_ipss", n42_flags,
                         notes="Val-18 N: fold-resolution sweep, perf+complexity oracle stack with 5x5 inner CV."),
        # N16_core_meta_dt removed: meta-DT killed (CC-V17-2).
        # N23_broad_banzhaf_clp_payoff removed: CLP+payoff now in base ref (CC-V17-3/V17-4).
        BenchmarkProfile("N21_broad_legacy_voting", "mnpo_broad_all", n21_flags,
                         notes="Val-18 N: broad set, legacy_voting (scale stress test)."),
        BenchmarkProfile("N22_broad_banzhaf", "mnpo_broad_all", n22_flags,
                         notes="Val-18 N: broad set, banzhaf weighting."),
    ]

    _val18_validate_method_sets(profiles, "validation18_mnpo")

    return _val18_build_jobs(
        run_family="val18_mnpo",
        profiles=profiles,
        dataset_ids=validation_ids,
        seeds=seeds,
        dataset_shards=dataset_shards,
        runtime_est=runtime_est,
    )


def _balanced_shard_assign_validation18_mnpo_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    return _balanced_shard_assign_profile_bundles(
        jobs, num_shards,
        run_family="val18_mnpo",
        profile_ids=VALIDATION18_MNPO_PROFILE_MANIFEST.keys(),
    )


# ---- Family D + P + S(D/P): Stage Matrix (DF/batch/prefilter/folding/gating/shadow) ----

VALIDATION18_STAGE_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    # Family D: Distribution-Fitting / Ordering / Batch / Fallback
    "D00_df_off_after": {"anchor": "yes", "contrast_ref": "D01_df_simple_after", "effect": "no_df_reference"},
    "D01_df_simple_after": {"anchor": "yes", "contrast_ref": "D00_df_off_after", "effect": "df_simple"},
    "D02_df_cvm_after": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "df_cvm"},
    "D03_df_ks_after": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "df_ks"},
    "D04_df_bic_after": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "df_bic"},
    "D05_df_aic_after": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "df_aic"},
    "D06_df_aicc_after": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "df_aicc"},
    "D07_df_cv_after": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "df_cv"},
    "D08_df_cv_loglik_after": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "df_cv_loglik"},
    "D09_df_crps_after": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "df_crps"},
    "D10_df_mnpo_after": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "df_mnpo_oracle"},
    "D11_df_simple_before": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "df_order_before"},
    "D12_df_mnpo_before": {"anchor": "no", "contrast_ref": "D10_df_mnpo_after", "effect": "df_mnpo_order_before"},
    "D13_batch_combat_source": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "batch_combat_source"},
    "D14_batch_combat_kmeans2": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "batch_combat_pseudo"},
    "D15_batch_cdf_center": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "batch_cdf_center"},
    "D16_multimodal_none": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "multimodal_off"},
    "D17_multimodal_gmm": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "multimodal_gmm"},
    "D18_multimodal_rank": {"anchor": "no", "contrast_ref": "D01_df_simple_after", "effect": "multimodal_rank_transform"},
    # Family P: Prefilter / Folding / Gating / Failure-Mode
    "P01_ref_scaffold": {"anchor": "yes", "contrast_ref": "P02_prefilter_union_off", "effect": "scaffold_reference"},
    "P02_prefilter_union_off": {"anchor": "no", "contrast_ref": "P01_ref_scaffold", "effect": "prefilter_union_off"},
    "P03_bh_off": {"anchor": "no", "contrast_ref": "P01_ref_scaffold", "effect": "bh_prefilter_off"},
    "P04_varfloor_off": {"anchor": "no", "contrast_ref": "P01_ref_scaffold", "effect": "variance_floor_off"},
    "P05_rank_prefilter_off": {"anchor": "no", "contrast_ref": "P01_ref_scaffold", "effect": "rank_prefilter_off"},
    "P06_screen_none": {"anchor": "no", "contrast_ref": "P01_ref_scaffold", "effect": "screening_off"},
    "P07_screen_evalue": {"anchor": "no", "contrast_ref": "P06_screen_none", "effect": "screening_evalue"},
    "P08_screen_stir": {"anchor": "no", "contrast_ref": "P06_screen_none", "effect": "screening_stir"},
    "P09_fold_none": {"anchor": "no", "contrast_ref": "P01_ref_scaffold", "effect": "folding_off"},
    "P10_fold_pls_da": {"anchor": "no", "contrast_ref": "P09_fold_none", "effect": "folding_pls_da"},
    "P11_fold_rff": {"anchor": "no", "contrast_ref": "P09_fold_none", "effect": "folding_rff"},
    "P12_fold_tensor_sketch": {"anchor": "no", "contrast_ref": "P09_fold_none", "effect": "folding_tensor_sketch"},
    "P13_regime_off": {"anchor": "no", "contrast_ref": "P01_ref_scaffold", "effect": "regime_gating_off"},
    "P14_lowpn_fast_filter": {"anchor": "no", "contrast_ref": "P01_ref_scaffold", "effect": "lowpn_fast_filter"},
    "P15_lowpn_all_features": {"anchor": "no", "contrast_ref": "P01_ref_scaffold", "effect": "lowpn_all_features"},
    "P16_extreme_multiclass_off": {"anchor": "no", "contrast_ref": "P01_ref_scaffold", "effect": "extreme_multiclass_off"},
    # P17_multiomics_mb_plsda removed: multiomics adapter killed (CC-V17-1).
    # P18_multiomics_mint removed: multiomics adapter killed (CC-V17-1).
    # Shadow profiles (D/P extension)
    # S01_diablo_blocks removed: multiomics adapter killed (CC-V17-1).
    "S02_batch_combat_seq": {"anchor": "no", "contrast_ref": "D13_batch_combat_source", "effect": "shadow_combat_seq"},
    "S03_cpss_irp_bounded": {"anchor": "no", "contrast_ref": "P01_ref_scaffold", "effect": "shadow_cpss"},
}


def build_jobs_validation18_stage(
    *,
    dataset_shards: int = 7,
    val17_root: Optional[Path] = None,
) -> List[Job]:
    """Val-18 Families D+P+S(D/P): stage/scaffold matrix x 24 diagnostic datasets x 5 seeds."""
    benchmark_datasets, _ = _load_benchmark_registry()
    if val17_root is None:
        val17_root = REPO_ROOT / "run_artifacts" / "validation-17"
    val17_runtime = _load_runtime_hints_from_summaries(val17_root, phase_tag="val17")
    validation_ids = list(VAL18_DIAG24)
    runtime_est = _val18_runtime_estimate(validation_ids, benchmark_datasets, val17_runtime)

    seeds = [11, 23, 37, 42, 59]  # 5 seeds
    ref = _val18_ref_flags()
    fs_method_set = "mnpo_v14_core_plus_ipss"

    # ---- D family helper: ref + override dist_criterion ----
    def _df_flags(criterion: str, position: str = "after_fs") -> Tuple[str, ...]:
        flags = list(ref)
        # Replace dist_criterion.
        try:
            idx = flags.index("simple")
            # Find the preceding --dist-criterion.
            if idx > 0 and flags[idx - 1] == "--dist-criterion":
                flags[idx] = criterion
        except ValueError:
            flags.extend(["--dist-criterion", criterion])
        # Replace stage position if needed.
        if position != "after_fs":
            try:
                idx = flags.index("after_fs")
                flags[idx] = position
            except ValueError:
                pass
        return tuple(flags)

    # D00: DF disabled.
    d00_flags = (*ref, "--disable-cdf-transform")
    # D01: simple after (default ref).
    d01_flags = ref
    # D02-D09: various DF criteria.
    d02_flags = _df_flags("cvm_p")
    d03_flags = _df_flags("ks_p")
    d04_flags = _df_flags("bic")
    d05_flags = _df_flags("aic")
    d06_flags = _df_flags("aicc")
    d07_flags = _df_flags("cv")
    d08_flags = _df_flags("cv_loglik")
    d09_flags = _df_flags("crps")
    d10_flags = _df_flags("mnpo_oracle")
    d11_flags = _df_flags("simple", "before_fs")
    d12_flags = _df_flags("mnpo_oracle", "before_fs")
    # D13: batch combat with source labels.
    d13_flags = (*ref, "--batch-correction", "combat")
    # D14: batch combat pseudo-batches (kmeans).
    d14_flags = (*ref, "--batch-correction", "combat", "--batch-label-policy", "kmeans2")
    # D15: batch cdf_center.
    d15_flags = (*ref, "--batch-correction", "cdf_center")
    # D16-D18: multimodal fallback.
    # NOTE: D16 originally used bare ref (no explicit --df-multimodal-fallback),
    # which silently defaulted to gmm — same as D17.  Fixed 2026-03-15 to
    # explicitly set --df-multimodal-fallback none so D16 is a true control.
    d16_flags = (*ref, "--df-multimodal-fallback", "none")
    d17_flags = (*ref, "--df-multimodal-fallback", "gmm")
    d18_flags = (*ref, "--df-multimodal-fallback", "rank_transform")

    # ---- P family flags ----
    # P01: full ref scaffold.
    p01_flags = ref
    # P02: prefilter union off.
    p02_flags_list = [f for f in ref if f != "--prefilter-union-enabled"]
    p02_flags = tuple(p02_flags_list)
    # P03: BH ttest off.
    p03_flags_list = [f for f in ref if f != "--prefilter-bh-ttest-enabled"]
    p03_flags = tuple(p03_flags_list)
    # P04: variance floor off.
    p04_flags_list = [f for f in ref if f != "--prefilter-variance-floor-enabled"]
    p04_flags = tuple(p04_flags_list)
    # P05: rank prefilter off.
    p05_flags = (*ref, "--disable-rank-prefilter")
    # P06: screening off.
    p06_flags_list = [f for f in ref if f not in ("--screening-enabled",)]
    p06_flags = tuple(p06_flags_list)
    # P07: screening evalue (should already be in ref, but make explicit).
    p07_flags = ref  # ref has --screening-method evalue.
    # P08: screening stir.
    p08_flags_list = list(ref)
    try:
        idx = p08_flags_list.index("evalue")
        if idx > 0 and p08_flags_list[idx - 1] == "--screening-method":
            p08_flags_list[idx] = "stir"
    except ValueError:
        pass
    p08_flags = tuple(p08_flags_list)
    # P09: no folding.
    p09_flags_list = list(ref)
    try:
        idx = p09_flags_list.index("pls_da")
        if idx > 0 and p09_flags_list[idx - 1] == "--folding-method":
            p09_flags_list[idx] = "none"
    except ValueError:
        pass
    p09_flags = tuple(p09_flags_list)
    # P10: folding pls_da (already in ref).
    p10_flags = ref
    # P11: folding rff.
    p11_flags_list = list(ref)
    try:
        idx = p11_flags_list.index("pls_da")
        if idx > 0 and p11_flags_list[idx - 1] == "--folding-method":
            p11_flags_list[idx] = "rff"
    except ValueError:
        pass
    p11_flags = tuple(p11_flags_list)
    # P12: folding tensor_sketch.
    p12_flags_list = list(ref)
    try:
        idx = p12_flags_list.index("pls_da")
        if idx > 0 and p12_flags_list[idx - 1] == "--folding-method":
            p12_flags_list[idx] = "tensor_sketch"
    except ValueError:
        pass
    p12_flags = tuple(p12_flags_list)
    # P13: regime gating off.
    p13_flags = _remove_prefixed_flags(ref, "--regime-gating")
    # P14: low-p/n fast univariate filter.
    p14_flags_list = list(ref)
    try:
        idx = p14_flags_list.index("0")
        if idx > 0 and p14_flags_list[idx - 1] == "--regime-gating-low-p-over-n-threshold":
            p14_flags_list[idx] = "50"
    except ValueError:
        pass
    p14_flags = (*tuple(p14_flags_list), "--regime-gating-low-p-over-n-mode", "fast_univariate_filter")
    # P15: low-p/n all features.
    p15_flags_list = list(ref)
    try:
        idx = p15_flags_list.index("0")
        if idx > 0 and p15_flags_list[idx - 1] == "--regime-gating-low-p-over-n-threshold":
            p15_flags_list[idx] = "50"
    except ValueError:
        pass
    p15_flags = (*tuple(p15_flags_list), "--regime-gating-low-p-over-n-mode", "all_features")
    # P16: extreme multiclass gate off.
    p16_flags_list = [
        f for f in ref
        if f not in ("--regime-gating-extreme-multiclass-enabled",)
    ]
    p16_flags = tuple(p16_flags_list)
    # P17/P18 multiomics profiles removed: multiomics adapter killed (CC-V17-1).

    # ---- Shadow profiles ----
    # S01_diablo_blocks removed: multiomics adapter killed (CC-V17-1).
    # S02: ComBat-seq.
    s02_flags = (*ref, "--batch-correction", "combat_seq")
    # S03: CPSS overlay on iterative_redundancy_pruning_bounded.
    s03_flags = (*ref, "--enable-fs-iterative-pruning-bounded-cpss-overlay")

    profiles: List[BenchmarkProfile] = [
        # D family
        BenchmarkProfile("D00_df_off_after", fs_method_set, d00_flags, notes="Val-18 D: DF disabled, after_fs."),
        BenchmarkProfile("D01_df_simple_after", fs_method_set, d01_flags, notes="Val-18 D: dist_criterion=simple, after_fs."),
        BenchmarkProfile("D02_df_cvm_after", fs_method_set, d02_flags, notes="Val-18 D: dist_criterion=cvm_p."),
        BenchmarkProfile("D03_df_ks_after", fs_method_set, d03_flags, notes="Val-18 D: dist_criterion=ks_p."),
        BenchmarkProfile("D04_df_bic_after", fs_method_set, d04_flags, notes="Val-18 D: dist_criterion=bic."),
        BenchmarkProfile("D05_df_aic_after", fs_method_set, d05_flags, notes="Val-18 D: dist_criterion=aic."),
        BenchmarkProfile("D06_df_aicc_after", fs_method_set, d06_flags, notes="Val-18 D: dist_criterion=aicc."),
        BenchmarkProfile("D07_df_cv_after", fs_method_set, d07_flags, notes="Val-18 D: dist_criterion=cv."),
        BenchmarkProfile("D08_df_cv_loglik_after", fs_method_set, d08_flags, notes="Val-18 D: dist_criterion=cv_loglik."),
        BenchmarkProfile("D09_df_crps_after", fs_method_set, d09_flags, notes="Val-18 D: dist_criterion=crps."),
        BenchmarkProfile("D10_df_mnpo_after", fs_method_set, d10_flags, notes="Val-18 D: dist_criterion=mnpo_oracle, after_fs."),
        BenchmarkProfile("D11_df_simple_before", fs_method_set, d11_flags, notes="Val-18 D: simple, before_fs (order control)."),
        BenchmarkProfile("D12_df_mnpo_before", fs_method_set, d12_flags, notes="Val-18 D: mnpo_oracle, before_fs."),
        BenchmarkProfile("D13_batch_combat_source", fs_method_set, d13_flags, notes="Val-18 D: batch_correction=combat, source labels."),
        BenchmarkProfile("D14_batch_combat_kmeans2", fs_method_set, d14_flags, notes="Val-18 D: batch_correction=combat, pseudo-batches."),
        BenchmarkProfile("D15_batch_cdf_center", fs_method_set, d15_flags, notes="Val-18 D: batch_correction=cdf_center."),
        BenchmarkProfile("D16_multimodal_none", fs_method_set, d16_flags, notes="Val-18 D: multimodal fallback off."),
        BenchmarkProfile("D17_multimodal_gmm", fs_method_set, d17_flags, notes="Val-18 D: multimodal_fallback=gmm."),
        BenchmarkProfile("D18_multimodal_rank", fs_method_set, d18_flags, notes="Val-18 D: multimodal_fallback=rank_transform."),
        # P family
        BenchmarkProfile("P01_ref_scaffold", fs_method_set, p01_flags, notes="Val-18 P: full reference scaffold."),
        BenchmarkProfile("P02_prefilter_union_off", fs_method_set, p02_flags, notes="Val-18 P: prefilter union disabled."),
        BenchmarkProfile("P03_bh_off", fs_method_set, p03_flags, notes="Val-18 P: BH ttest prefilter disabled."),
        BenchmarkProfile("P04_varfloor_off", fs_method_set, p04_flags, notes="Val-18 P: variance floor disabled."),
        BenchmarkProfile("P05_rank_prefilter_off", fs_method_set, p05_flags, notes="Val-18 P: rank prefilter disabled."),
        BenchmarkProfile("P06_screen_none", fs_method_set, p06_flags, notes="Val-18 P: screening off."),
        BenchmarkProfile("P07_screen_evalue", fs_method_set, p07_flags, notes="Val-18 P: screening=evalue."),
        BenchmarkProfile("P08_screen_stir", fs_method_set, p08_flags, notes="Val-18 P: screening=stir."),
        BenchmarkProfile("P09_fold_none", fs_method_set, p09_flags, notes="Val-18 P: no folding."),
        BenchmarkProfile("P10_fold_pls_da", fs_method_set, p10_flags, notes="Val-18 P: folding_method=pls_da."),
        BenchmarkProfile("P11_fold_rff", fs_method_set, p11_flags, notes="Val-18 P: folding_method=rff."),
        BenchmarkProfile("P12_fold_tensor_sketch", fs_method_set, p12_flags, notes="Val-18 P: folding_method=tensor_sketch."),
        BenchmarkProfile("P13_regime_off", fs_method_set, p13_flags, notes="Val-18 P: regime gating disabled."),
        BenchmarkProfile("P14_lowpn_fast_filter", fs_method_set, p14_flags, notes="Val-18 P: low-p/n bypass=fast_univariate_filter."),
        BenchmarkProfile("P15_lowpn_all_features", fs_method_set, p15_flags, notes="Val-18 P: low-p/n bypass=all_features."),
        BenchmarkProfile("P16_extreme_multiclass_off", fs_method_set, p16_flags, notes="Val-18 P: extreme-multiclass gate disabled."),
        # P17/P18 multiomics profiles removed: multiomics adapter killed (CC-V17-1).
        # Shadow
        # S01_diablo_blocks removed: multiomics adapter killed (CC-V17-1).
        BenchmarkProfile("S02_batch_combat_seq", fs_method_set, s02_flags, notes="Val-18 shadow: ComBat-seq batch correction."),
        BenchmarkProfile("S03_cpss_irp_bounded", "singleton_iterative_redundancy_pruning_bounded", s03_flags,
                         notes="Val-18 shadow: CPSS overlay on iterative_redundancy_pruning_bounded."),
    ]

    _val18_validate_method_sets(profiles, "validation18_stage")

    return _val18_build_jobs(
        run_family="val18_stage",
        profiles=profiles,
        dataset_ids=validation_ids,
        seeds=seeds,
        dataset_shards=dataset_shards,
        runtime_est=runtime_est,
    )


def _balanced_shard_assign_validation18_stage_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    return _balanced_shard_assign_profile_bundles(
        jobs, num_shards,
        run_family="val18_stage",
        profile_ids=VALIDATION18_STAGE_PROFILE_MANIFEST.keys(),
    )


# ---- Family C + S(C): Classifier / TabPFN / Conformal Matrix ----

VALIDATION18_CLASSIFIERS_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {}
# Add C-ONLY-{classifier} for each classifier.
for _clf in VAL18_CLASSIFIER_UNIVERSE:
    VALIDATION18_CLASSIFIERS_PROFILE_MANIFEST[f"C_ONLY_{_clf}"] = {
        "anchor": "no",
        "contrast_ref": "C_ONLY_dlda",
        "effect": f"classifier_only_{_clf}",
    }
del _clf
# Add pool and conformal profiles.
VALIDATION18_CLASSIFIERS_PROFILE_MANIFEST.update({
    "C01_pool_legacy_core": {"anchor": "yes", "contrast_ref": "C03_pool_mnpo_core", "effect": "pool_legacy"},
    "C02_pool_legacy_plus_tabpfn": {"anchor": "no", "contrast_ref": "C01_pool_legacy_core", "effect": "pool_legacy_tabpfn"},
    "C03_pool_mnpo_core": {"anchor": "no", "contrast_ref": "C01_pool_legacy_core", "effect": "pool_mnpo"},
    "C04_pool_mnpo_plus_tabpfn": {"anchor": "no", "contrast_ref": "C03_pool_mnpo_core", "effect": "pool_mnpo_tabpfn"},
    "C05_conformal_off": {"anchor": "no", "contrast_ref": "C01_pool_legacy_core", "effect": "conformal_off"},
    "C06_conformal_split": {"anchor": "no", "contrast_ref": "C05_conformal_off", "effect": "conformal_split"},
    "C07_conformal_aps": {"anchor": "no", "contrast_ref": "C05_conformal_off", "effect": "conformal_aps"},
    "C08_conformal_raps": {"anchor": "no", "contrast_ref": "C05_conformal_off", "effect": "conformal_raps"},
    "C09_conformal_cross": {"anchor": "no", "contrast_ref": "C05_conformal_off", "effect": "conformal_cross"},
    "C10_only_tabpfn_full64": {"anchor": "no", "contrast_ref": "C_ONLY_tabpfn", "effect": "tabpfn_only_full64", "panel": "full64", "execution_lane": "tabpfn"},
    "C11_pool_legacy_plus_tabpfn_full64": {"anchor": "no", "contrast_ref": "C02_pool_legacy_plus_tabpfn", "effect": "pool_legacy_tabpfn_full64", "panel": "full64", "execution_lane": "tabpfn"},
    "C12_pool_mnpo_plus_tabpfn_full64": {"anchor": "no", "contrast_ref": "C04_pool_mnpo_plus_tabpfn", "effect": "pool_mnpo_tabpfn_full64", "panel": "full64", "execution_lane": "tabpfn"},
    # Shadow S04
    "S04_rp_ensemble": {"anchor": "no", "contrast_ref": "C_ONLY_dlda", "effect": "shadow_rp_ensemble"},
    # Shadow S05-S07: HDLSS bias-correction candidates
    "S05_dbda": {"anchor": "no", "contrast_ref": "C_ONLY_dlda", "effect": "shadow_dbda"},
    "S06_gqda": {"anchor": "no", "contrast_ref": "C_ONLY_dlda", "effect": "shadow_gqda"},
    "S07_bc_svm_linear": {"anchor": "no", "contrast_ref": "C_ONLY_svm_linear", "effect": "shadow_bc_svm_linear"},
    # Shadow S08: Sparse Group Lasso NN (Yang 2020)
    "S08_sglnn": {"anchor": "no", "contrast_ref": "C_ONLY_dlda", "effect": "shadow_sglnn"},
})


def build_jobs_validation18_classifiers(
    *,
    dataset_shards: int = 7,
    val17_root: Optional[Path] = None,
) -> List[Job]:
    """Val-18 Family C + S04: CPU classifier sweep plus TabPFN-only reruns."""
    benchmark_datasets, _ = _load_benchmark_registry()
    if val17_root is None:
        val17_root = REPO_ROOT / "run_artifacts" / "validation-17"
    val17_runtime = _load_runtime_hints_from_summaries(val17_root, phase_tag="val17")
    diag24_ids = list(VAL18_DIAG24)
    full64_ids = list(VAL18_FULL64)
    runtime_est_diag24 = _val18_runtime_estimate(diag24_ids, benchmark_datasets, val17_runtime)
    runtime_est_full64 = _val18_runtime_estimate(full64_ids, benchmark_datasets, val17_runtime)

    seeds = [11, 23, 37, 42, 59]  # 5 seeds
    ref = _val18_ref_flags()
    fs_method_set = "mnpo_v14_core_plus_ipss"
    cpu_job_params = {
        "dataset_panel": "diag24",
        "execution_lane": "cpu",
        "preferred_hosts": list(VAL18_CPU_HOSTS),
    }
    tabpfn_diag_job_params = {
        "dataset_panel": "diag24",
        "execution_lane": "tabpfn",
        "preferred_hosts": list(VAL18_GPU_HOSTS),
    }
    tabpfn_full64_job_params = {
        "dataset_panel": "full64",
        "execution_lane": "tabpfn",
        "preferred_hosts": list(VAL18_GPU_HOSTS),
    }

    # Fixed upstream scaffold for C-ONLY: ref flags + fixed single classifier.
    legacy_clf_base: Tuple[str, ...] = (
        *ref,
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
    )

    # Legacy classifier pool models (core 8).
    legacy_pool_models: Tuple[str, ...] = (
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
    )

    # MNPO hybrid classifier flags.
    mnpo_clf_base: Tuple[str, ...] = (
        *ref,
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
    )

    diag24_profiles: List[BenchmarkProfile] = []
    full64_tabpfn_profiles: List[BenchmarkProfile] = []

    # C-ONLY-{classifier} profiles.
    for clf in VAL18_CLASSIFIER_UNIVERSE:
        clf_flags = (*legacy_clf_base, "--model-candidates", clf, "--model-cv-runtime-max-candidates", "1")
        if clf == "tabpfn":
            clf_flags = (*clf_flags, "--include-tabpfn-model")
            diag24_profiles.append(BenchmarkProfile(
                f"C_ONLY_{clf}", fs_method_set, clf_flags,
                weight_mult=1.5,
                notes=f"Val-18 C: single classifier={clf} on fixed upstream scaffold.",
                job_params=tabpfn_diag_job_params,
            ))
            full64_tabpfn_profiles.append(BenchmarkProfile(
                "C10_only_tabpfn_full64", fs_method_set, clf_flags,
                weight_mult=1.5,
                notes="Val-18 C: FULL64 rerun of the single-classifier TabPFN scaffold.",
                job_params=tabpfn_full64_job_params,
            ))
            continue
        diag24_profiles.append(BenchmarkProfile(
            f"C_ONLY_{clf}", fs_method_set, clf_flags,
            notes=f"Val-18 C: single classifier={clf} on fixed upstream scaffold.",
            job_params=cpu_job_params,
        ))

    # C01: legacy pool core.
    c01_flags = (*legacy_clf_base, "--model-candidates", *legacy_pool_models, "--model-cv-runtime-max-candidates", "8")
    diag24_profiles.append(BenchmarkProfile("C01_pool_legacy_core", fs_method_set, c01_flags,
                                            notes="Val-18 C: legacy CV classifier pool (core 8).",
                                            job_params=cpu_job_params))

    # C02: legacy pool + TabPFN.
    c02_flags = (*legacy_clf_base, "--model-candidates", *legacy_pool_models, "tabpfn",
                 "--include-tabpfn-model", "--model-cv-runtime-max-candidates", "9")
    diag24_profiles.append(BenchmarkProfile("C02_pool_legacy_plus_tabpfn", fs_method_set, c02_flags,
                                            weight_mult=1.5,
                                            notes="Val-18 C: legacy pool + TabPFN.",
                                            job_params=tabpfn_diag_job_params))
    full64_tabpfn_profiles.append(BenchmarkProfile("C11_pool_legacy_plus_tabpfn_full64", fs_method_set, c02_flags,
                                                   weight_mult=1.5,
                                                   notes="Val-18 C: FULL64 rerun of the legacy pool + TabPFN profile.",
                                                   job_params=tabpfn_full64_job_params))

    # C03: MNPO pool core.
    c03_flags = (*mnpo_clf_base, "--model-candidates", *legacy_pool_models, "--model-cv-runtime-max-candidates", "8")
    diag24_profiles.append(BenchmarkProfile("C03_pool_mnpo_core", fs_method_set, c03_flags,
                                            notes="Val-18 C: MNPO hybrid classifier pool (core 8).",
                                            job_params=cpu_job_params))

    # C04: MNPO pool + TabPFN.
    c04_flags = (*mnpo_clf_base, "--model-candidates", *legacy_pool_models, "tabpfn",
                 "--include-tabpfn-model", "--model-cv-runtime-max-candidates", "9")
    diag24_profiles.append(BenchmarkProfile("C04_pool_mnpo_plus_tabpfn", fs_method_set, c04_flags,
                                            weight_mult=1.5,
                                            notes="Val-18 C: MNPO hybrid pool + TabPFN.",
                                            job_params=tabpfn_diag_job_params))
    full64_tabpfn_profiles.append(BenchmarkProfile("C12_pool_mnpo_plus_tabpfn_full64", fs_method_set, c04_flags,
                                                   weight_mult=1.5,
                                                   notes="Val-18 C: FULL64 rerun of the MNPO hybrid pool + TabPFN profile.",
                                                   job_params=tabpfn_full64_job_params))

    # Conformal profiles (C05-C09).
    # C05: conformal off.
    conformal_off_flags = _remove_prefixed_flags(ref, "--classifier-conformal")
    conformal_off_flags = _remove_flag(conformal_off_flags, "--enable-classifier-conformal")
    conformal_off_flags = (*conformal_off_flags, "--classifier-selection-mode", "legacy",
                           "--classification-backend", "sklearn",
                           "--model-candidates", *legacy_pool_models, "--model-cv-runtime-max-candidates", "8")
    diag24_profiles.append(BenchmarkProfile("C05_conformal_off", fs_method_set, conformal_off_flags,
                                            notes="Val-18 C: conformal off (uncertainty baseline).",
                                            job_params=cpu_job_params))

    for conf_id, conf_method in [
        ("C06_conformal_split", "split"),
        ("C07_conformal_aps", "aps"),
        ("C08_conformal_raps", "raps"),
        ("C09_conformal_cross", "cross"),
    ]:
        conf_flags = (*legacy_clf_base,
                      "--model-candidates", *legacy_pool_models, "--model-cv-runtime-max-candidates", "8",
                      "--enable-classifier-conformal",
                      "--classifier-conformal-method", conf_method,
                      "--classifier-conformal-alpha", "0.10",
                      "--classifier-conformal-calibration-fraction", "0.25",
                      "--classifier-conformal-min-calibration", "20")
        diag24_profiles.append(BenchmarkProfile(conf_id, fs_method_set, conf_flags,
                                                notes=f"Val-18 C: conformal method={conf_method}.",
                                                job_params=cpu_job_params))

    # S04: rp_ensemble shadow (same as C-ONLY but explicit shadow).
    s04_flags = (*legacy_clf_base, "--model-candidates", "rp_ensemble", "--model-cv-runtime-max-candidates", "1")
    diag24_profiles.append(BenchmarkProfile("S04_rp_ensemble", fs_method_set, s04_flags,
                                            notes="Val-18 shadow: C-ONLY-rp_ensemble.",
                                            job_params=cpu_job_params))

    # S05-S07: HDLSS bias-correction shadow profiles (Aoshima & Yata).
    for shadow_id, shadow_clf in [
        ("S05_dbda", "dbda"),
        ("S06_gqda", "gqda"),
        ("S07_bc_svm_linear", "bc_svm_linear"),
    ]:
        s_flags = (*legacy_clf_base, "--model-candidates", shadow_clf, "--model-cv-runtime-max-candidates", "1")
        diag24_profiles.append(BenchmarkProfile(shadow_id, fs_method_set, s_flags,
                                                notes=f"Val-18 shadow: C-ONLY-{shadow_clf}.",
                                                job_params=cpu_job_params))

    # S08: Sparse Group Lasso NN shadow profile (Yang 2020).
    s08_flags = (*legacy_clf_base, "--model-candidates", "sglnn", "--model-cv-runtime-max-candidates", "1")
    diag24_profiles.append(BenchmarkProfile("S08_sglnn", fs_method_set, s08_flags,
                                            notes="Val-18 shadow: C-ONLY-sglnn (Yang 2020, Ch. 3).",
                                            job_params=cpu_job_params))

    all_profiles = [*diag24_profiles, *full64_tabpfn_profiles]
    _val18_validate_method_sets(all_profiles, "validation18_classifiers")

    jobs = _val18_build_jobs(
        run_family="val18_classifiers",
        profiles=diag24_profiles,
        dataset_ids=diag24_ids,
        seeds=seeds,
        dataset_shards=dataset_shards,
        runtime_est=runtime_est_diag24,
    )
    jobs.extend(
        _val18_build_jobs(
            run_family="val18_classifiers",
            profiles=full64_tabpfn_profiles,
            dataset_ids=full64_ids,
            seeds=seeds,
            dataset_shards=dataset_shards,
            runtime_est=runtime_est_full64,
        )
    )
    return jobs


def _balanced_shard_assign_validation18_classifiers_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Keep CPU classifier work separate from TabPFN-only arch-ml lanes."""
    if int(num_shards) <= 1:
        return _balanced_shard_assign_profile_bundles(
            jobs, num_shards,
            run_family="val18_classifiers",
            profile_ids=VALIDATION18_CLASSIFIERS_PROFILE_MANIFEST.keys(),
        )

    grouped: Dict[Tuple[str, str, str], List[Job]] = {}
    ungrouped: List[Job] = []
    for job in jobs:
        parts = str(job.job_id).split("/")
        if len(parts) != 3 or parts[0] != "val18_classifiers":
            ungrouped.append(job)
            continue
        params = dict(job.params or {})
        lane = str(params.get("execution_lane", "") or "").strip().lower()
        panel = str(params.get("dataset_panel", "") or "").strip().lower()
        if not lane:
            lane = "tabpfn" if "tabpfn" in str(job.job_id).lower() else "cpu"
        if not panel:
            panel = "full64" if str(parts[1]).endswith("_full64") else "diag24"
        grouped.setdefault((lane, panel, str(parts[2])), []).append(job)

    cpu_bundles: List[Tuple[float, List[Job]]] = []
    tabpfn_bundles: List[Tuple[float, List[Job]]] = []
    for (lane, _panel, _part), bundle_jobs in grouped.items():
        bundle_weight = sum(float(j.weight) for j in bundle_jobs)
        record = (bundle_weight, sorted(bundle_jobs, key=lambda j: j.job_id))
        if lane == "tabpfn":
            tabpfn_bundles.append(record)
        else:
            cpu_bundles.append(record)

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    cpu_weight = sum(float(w) for w, _ in cpu_bundles)
    tabpfn_weight = sum(float(w) for w, _ in tabpfn_bundles)
    if cpu_bundles and tabpfn_bundles:
        total_weight = float(cpu_weight + tabpfn_weight)
        tabpfn_shards = int(round(float(num_shards) * float(tabpfn_weight / total_weight)))
        tabpfn_shards = max(1, min(int(num_shards) - 1, tabpfn_shards))
        cpu_shards = int(num_shards) - tabpfn_shards
        cpu_shard_ids = list(range(1, cpu_shards + 1))
        tabpfn_shard_ids = list(range(cpu_shards + 1, int(num_shards) + 1))
    elif tabpfn_bundles:
        cpu_shard_ids = []
        tabpfn_shard_ids = list(range(1, int(num_shards) + 1))
    else:
        cpu_shard_ids = list(range(1, int(num_shards) + 1))
        tabpfn_shard_ids = []

    def _assign(bundle_items: Sequence[Tuple[float, List[Job]]], shard_ids: Sequence[int]) -> None:
        if not shard_ids:
            return
        for bundle_weight, bundle_jobs in sorted(bundle_items, key=lambda x: float(x[0]), reverse=True):
            target = min(((sid, totals[sid]) for sid in shard_ids), key=lambda kv: kv[1])[0]
            for bundle_job in bundle_jobs:
                shards[target].append(bundle_job.job_id)
            totals[target] += float(bundle_weight)

    _assign(cpu_bundles, cpu_shard_ids)
    _assign(tabpfn_bundles, tabpfn_shard_ids)

    if ungrouped:
        fallback_ids = cpu_shard_ids or tabpfn_shard_ids or list(range(1, int(num_shards) + 1))
        for job in sorted(ungrouped, key=lambda j: float(j.weight), reverse=True):
            target = min(((sid, totals[sid]) for sid in fallback_ids), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


# ---- Validation-19: Added-Classifier Expansion Campaign ----

VALIDATION19_CLASSIFIERS_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {}
for _clf in VAL19_ADDED_CLASSIFIERS:
    VALIDATION19_CLASSIFIERS_PROFILE_MANIFEST[f"C_ONLY_{_clf}"] = {
        "anchor": "no",
        "contrast_ref": "C_ONLY_dlda",
        "effect": f"added_classifier_singleton_{_clf}",
        "panel": "diag24",
    }
del _clf

VALIDATION19_CLASSIFIERS_PROFILE_MANIFEST.update({
    "V19_C01_old_regime_legacy_full64": {
        "anchor": "yes",
        "contrast_ref": "V19_C02_new_regime_legacy_full64",
        "effect": "old_regime_legacy_full64",
        "panel": "full64",
        "pool_snapshot_id": "old",
        "selector_mode": "legacy",
    },
    "V19_C02_new_regime_legacy_full64": {
        "anchor": "no",
        "contrast_ref": "V19_C01_old_regime_legacy_full64",
        "effect": "new_regime_legacy_full64",
        "panel": "full64",
        "pool_snapshot_id": "new",
        "selector_mode": "legacy",
    },
    "V19_C03_old_regime_mnpo_full64": {
        "anchor": "yes",
        "contrast_ref": "V19_C04_new_regime_mnpo_full64",
        "effect": "old_regime_mnpo_full64",
        "panel": "full64",
        "pool_snapshot_id": "old",
        "selector_mode": "mnpo_hybrid",
    },
    "V19_C04_new_regime_mnpo_full64": {
        "anchor": "no",
        "contrast_ref": "V19_C03_old_regime_mnpo_full64",
        "effect": "new_regime_mnpo_full64",
        "panel": "full64",
        "pool_snapshot_id": "new",
        "selector_mode": "mnpo_hybrid",
    },
    "V19_C05_old_regime_mnpo_val18compat_full64": {
        "anchor": "no",
        "contrast_ref": "V19_C03_old_regime_mnpo_full64",
        "effect": "old_regime_mnpo_val18compat_full64",
        "panel": "full64",
        "pool_snapshot_id": "old",
        "selector_mode": "mnpo_hybrid",
        "oracle_behavior_profile": "val18_compat",
    },
    "V19_C06_new_regime_mnpo_val18compat_full64": {
        "anchor": "no",
        "contrast_ref": "V19_C04_new_regime_mnpo_full64",
        "effect": "new_regime_mnpo_val18compat_full64",
        "panel": "full64",
        "pool_snapshot_id": "new",
        "selector_mode": "mnpo_hybrid",
        "oracle_behavior_profile": "val18_compat",
    },
})


def build_jobs_validation19_classifiers(
    *,
    dataset_shards: int = 7,
    val17_root: Optional[Path] = None,
) -> List[Job]:
    """Val-19 required surface: 7 singleton diagnostics + 4 pool reruns + 2 oracle-compat controls."""
    benchmark_datasets, _ = _load_benchmark_registry()
    if val17_root is None:
        val17_root = REPO_ROOT / "run_artifacts" / "validation-17"
    val17_runtime = _load_runtime_hints_from_summaries(val17_root, phase_tag="val17")
    diag24_ids = list(VAL18_DIAG24)
    full64_ids = list(VAL18_FULL64)
    runtime_est_diag24 = _val18_runtime_estimate(diag24_ids, benchmark_datasets, val17_runtime)
    runtime_est_full64 = _val18_runtime_estimate(full64_ids, benchmark_datasets, val17_runtime)

    seeds = [11, 23, 37, 42, 59]
    ref = _val18_ref_flags()
    fs_method_set = "mnpo_v14_core_plus_ipss"

    cpu_diag_job_params = {
        "dataset_panel": "diag24",
        "execution_lane": "cpu",
        "preferred_hosts": list(VAL18_CPU_HOSTS),
    }
    full64_job_params = {
        "dataset_panel": "full64",
        "execution_lane": "cpu",
        "preferred_hosts": list(VAL18_CPU_HOSTS),
    }

    singleton_base: Tuple[str, ...] = (
        *ref,
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
    )
    legacy_pool_base: Tuple[str, ...] = (
        *ref,
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
    )
    mnpo_pool_base: Tuple[str, ...] = (
        *ref,
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classifier-oracle-behavior-profile", "current",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
    )
    mnpo_val18_compat_pool_base: Tuple[str, ...] = _replace_flag_value(
        mnpo_pool_base,
        "--classifier-oracle-behavior-profile",
        "val18_compat",
    )

    diag24_profiles: List[BenchmarkProfile] = []
    for clf in VAL19_ADDED_CLASSIFIERS:
        diag24_profiles.append(
            BenchmarkProfile(
                f"C_ONLY_{clf}",
                fs_method_set,
                (*singleton_base, "--model-candidates", clf, "--model-cv-runtime-max-candidates", "1"),
                notes=f"Val-19 singleton rerun for added classifier={clf} on the fixed Val-18 scaffold.",
                job_params=cpu_diag_job_params,
            )
        )

    old_pool_candidates = tuple(VAL19_HDLSS_MODERATE_CPU_OLD)
    new_pool_candidates = tuple(VAL19_HDLSS_MODERATE_CPU_NEW)
    max_old = str(len(old_pool_candidates))
    max_new = str(len(new_pool_candidates))

    full64_profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            "V19_C01_old_regime_legacy_full64",
            fs_method_set,
            (*legacy_pool_base, "--model-candidates", *old_pool_candidates, "--model-cv-runtime-max-candidates", max_old),
            notes="Val-19 matched full-catalog old-pool baseline under legacy classifier selection.",
            job_params={**full64_job_params, "pool_snapshot_id": "old", "selector_mode": "legacy"},
        ),
        BenchmarkProfile(
            "V19_C02_new_regime_legacy_full64",
            fs_method_set,
            (*legacy_pool_base, "--model-candidates", *new_pool_candidates, "--model-cv-runtime-max-candidates", max_new),
            notes="Val-19 matched full-catalog new-pool comparison under legacy classifier selection.",
            job_params={**full64_job_params, "pool_snapshot_id": "new", "selector_mode": "legacy"},
        ),
        BenchmarkProfile(
            "V19_C03_old_regime_mnpo_full64",
            fs_method_set,
            (*mnpo_pool_base, "--model-candidates", *old_pool_candidates, "--model-cv-runtime-max-candidates", max_old),
            notes="Val-19 matched full-catalog old-pool baseline under MNPO hybrid classifier selection.",
            job_params={
                **full64_job_params,
                "pool_snapshot_id": "old",
                "selector_mode": "mnpo_hybrid",
                "oracle_behavior_profile": "current",
            },
        ),
        BenchmarkProfile(
            "V19_C04_new_regime_mnpo_full64",
            fs_method_set,
            (*mnpo_pool_base, "--model-candidates", *new_pool_candidates, "--model-cv-runtime-max-candidates", max_new),
            notes="Val-19 matched full-catalog new-pool comparison under MNPO hybrid classifier selection.",
            job_params={
                **full64_job_params,
                "pool_snapshot_id": "new",
                "selector_mode": "mnpo_hybrid",
                "oracle_behavior_profile": "current",
            },
        ),
        BenchmarkProfile(
            "V19_C05_old_regime_mnpo_val18compat_full64",
            fs_method_set,
            (*mnpo_val18_compat_pool_base, "--model-candidates", *old_pool_candidates, "--model-cv-runtime-max-candidates", max_old),
            notes="Val-19 matched full-catalog old-pool control under val18-compatible MNPO classifier-oracle behavior.",
            job_params={
                **full64_job_params,
                "pool_snapshot_id": "old",
                "selector_mode": "mnpo_hybrid",
                "oracle_behavior_profile": "val18_compat",
            },
        ),
        BenchmarkProfile(
            "V19_C06_new_regime_mnpo_val18compat_full64",
            fs_method_set,
            (*mnpo_val18_compat_pool_base, "--model-candidates", *new_pool_candidates, "--model-cv-runtime-max-candidates", max_new),
            notes="Val-19 matched full-catalog new-pool control under val18-compatible MNPO classifier-oracle behavior.",
            job_params={
                **full64_job_params,
                "pool_snapshot_id": "new",
                "selector_mode": "mnpo_hybrid",
                "oracle_behavior_profile": "val18_compat",
            },
        ),
    ]

    all_profiles = [*diag24_profiles, *full64_profiles]
    _val18_validate_method_sets(all_profiles, "validation19_classifiers")

    jobs = _val18_build_jobs(
        run_family="val19_classifiers",
        profiles=diag24_profiles,
        dataset_ids=diag24_ids,
        seeds=seeds,
        dataset_shards=dataset_shards,
        runtime_est=runtime_est_diag24,
    )
    jobs.extend(
        _val18_build_jobs(
            run_family="val19_classifiers",
            profiles=full64_profiles,
            dataset_ids=full64_ids,
            seeds=seeds,
            dataset_shards=dataset_shards,
            runtime_est=runtime_est_full64,
        )
    )
    return jobs


def _balanced_shard_assign_validation19_classifiers_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-19 by panel-aware dataset bundles to keep matched surfaces aligned."""
    grouped: Dict[Tuple[str, str], List[Job]] = {}
    ungrouped: List[Job] = []
    for job in jobs:
        parts = str(job.job_id).split("/")
        if len(parts) != 3 or parts[0] != "val19_classifiers":
            ungrouped.append(job)
            continue
        panel = str(dict(job.params or {}).get("dataset_panel", "") or "").strip().lower() or "diag24"
        grouped.setdefault((panel, str(parts[2])), []).append(job)

    bundles: List[Tuple[float, List[Job]]] = []
    for bundle_jobs in grouped.values():
        bundle_weight = sum(float(j.weight) for j in bundle_jobs)
        bundles.append((bundle_weight, sorted(bundle_jobs, key=lambda j: j.job_id)))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for bundle_weight, bundle_jobs in sorted(bundles, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        for bundle_job in bundle_jobs:
            shards[target].append(bundle_job.job_id)
        totals[target] += float(bundle_weight)

    for job in sorted(ungrouped, key=lambda j: float(j.weight), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        shards[target].append(job.job_id)
        totals[target] += float(job.weight)

    return shards


# ---- Family E: Ensemble & Oracle Enhancements (Val-20) ----

VALIDATION20_ENSEMBLE_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "V20_E01_soft_voting_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_E07_all_enhancements_diag24",
        "effect": "soft_voting_only",
        "panel": "diag24",
    },
    "V20_E02_greedy_ensemble_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_E07_all_enhancements_diag24",
        "effect": "greedy_ensemble_only",
        "panel": "diag24",
    },
    "V20_E03_candidate_pruning_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_E07_all_enhancements_diag24",
        "effect": "candidate_pruning_only",
        "panel": "diag24",
    },
    "V20_E04_incumbent_racing_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_E07_all_enhancements_diag24",
        "effect": "incumbent_racing_only",
        "panel": "diag24",
    },
    "V20_E05_soft_greedy_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_E01_soft_voting_diag24",
        "effect": "soft_voting_plus_greedy",
        "panel": "diag24",
    },
    "V20_E06_pruning_incumbent_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_E03_candidate_pruning_diag24",
        "effect": "pruning_plus_incumbent",
        "panel": "diag24",
    },
    "V20_E07_all_enhancements_diag24": {
        "anchor": "yes",
        "contrast_ref": "V20_E01_soft_voting_diag24",
        "effect": "all_enhancements",
        "panel": "diag24",
    },
}


def build_jobs_validation20_ensemble(
    *,
    dataset_shards: int = 7,
    val17_root: Optional[Path] = None,
) -> List[Job]:
    """Val-20 Family E: ensemble and oracle enhancement ablation on DIAG24."""
    benchmark_datasets, _ = _load_benchmark_registry()
    if val17_root is None:
        val17_root = REPO_ROOT / "run_artifacts" / "validation-17"
    val17_runtime = _load_runtime_hints_from_summaries(val17_root, phase_tag="val17")
    diag24_ids = list(VAL18_DIAG24)
    runtime_est = _val18_runtime_estimate(diag24_ids, benchmark_datasets, val17_runtime)

    seeds = [11, 23, 37, 42, 59]
    ref = _val18_ref_flags()
    fs_method_set = "mnpo_v14_core_plus_ipss"

    cpu_job_params = {
        "dataset_panel": "diag24",
        "execution_lane": "cpu",
        "preferred_hosts": list(VAL18_CPU_HOSTS),
    }

    # Shared MNPO hybrid base flags for all E-profiles.
    mnpo_base: Tuple[str, ...] = (
        *ref,
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classifier-oracle-behavior-profile", "current",
        "--classifier-oracle-k", "3",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
    )

    profiles: List[BenchmarkProfile] = [
        # E01: B2 soft voting only
        BenchmarkProfile(
            "V20_E01_soft_voting_diag24", fs_method_set,
            (*mnpo_base, "--classifier-oracle-ensemble-voting-mode", "soft"),
            notes="Val-20 E01: soft voting with Nash weights (B2 only).",
            job_params=cpu_job_params,
        ),
        # E02: B1 greedy ensemble selection only
        BenchmarkProfile(
            "V20_E02_greedy_ensemble_diag24", fs_method_set,
            (*mnpo_base, "--enable-classifier-oracle-greedy-ensemble",
             "--classifier-oracle-greedy-ensemble-rounds", "10"),
            notes="Val-20 E02: greedy ensemble selection with replacement (B1 only).",
            job_params=cpu_job_params,
        ),
        # E03: B3 candidate pruning only
        BenchmarkProfile(
            "V20_E03_candidate_pruning_diag24", fs_method_set,
            (*mnpo_base, "--enable-classifier-oracle-candidate-pruning",
             "--classifier-oracle-candidate-pruning-threshold", "0.0"),
            notes="Val-20 E03: Troupe-style LOO marginal pruning (B3 only).",
            job_params=cpu_job_params,
        ),
        # E04: B8 incumbent-based early stopping only
        BenchmarkProfile(
            "V20_E04_incumbent_racing_diag24", fs_method_set,
            (*mnpo_base, "--enable-classifier-oracle-incumbent-early-stopping"),
            notes="Val-20 E04: incumbent-based early stopping in Hoeffding racing (B8 only).",
            job_params=cpu_job_params,
        ),
        # E05: B2+B1 combined
        BenchmarkProfile(
            "V20_E05_soft_greedy_diag24", fs_method_set,
            (*mnpo_base, "--classifier-oracle-ensemble-voting-mode", "soft",
             "--enable-classifier-oracle-greedy-ensemble",
             "--classifier-oracle-greedy-ensemble-rounds", "10"),
            notes="Val-20 E05: soft voting + greedy ensemble (B2+B1).",
            job_params=cpu_job_params,
        ),
        # E06: B3+B8 combined
        BenchmarkProfile(
            "V20_E06_pruning_incumbent_diag24", fs_method_set,
            (*mnpo_base, "--enable-classifier-oracle-candidate-pruning",
             "--classifier-oracle-candidate-pruning-threshold", "0.0",
             "--enable-classifier-oracle-incumbent-early-stopping"),
            notes="Val-20 E06: candidate pruning + incumbent racing (B3+B8).",
            job_params=cpu_job_params,
        ),
        # E07: All four enhancements combined
        BenchmarkProfile(
            "V20_E07_all_enhancements_diag24", fs_method_set,
            (*mnpo_base, "--classifier-oracle-ensemble-voting-mode", "soft",
             "--enable-classifier-oracle-greedy-ensemble",
             "--classifier-oracle-greedy-ensemble-rounds", "10",
             "--enable-classifier-oracle-candidate-pruning",
             "--classifier-oracle-candidate-pruning-threshold", "0.0",
             "--enable-classifier-oracle-incumbent-early-stopping"),
            notes="Val-20 E07: all four enhancement toggles (B1+B2+B3+B8).",
            job_params=cpu_job_params,
        ),
    ]

    _val18_validate_method_sets(profiles, "validation20_ensemble")

    return _val18_build_jobs(
        run_family="val20_ensemble",
        profiles=profiles,
        dataset_ids=diag24_ids,
        seeds=seeds,
        dataset_shards=dataset_shards,
        runtime_est=runtime_est,
    )


def _balanced_shard_assign_validation20_ensemble_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """CPU-only shard assignment for Val-20 Family E."""
    return _balanced_shard_assign_profile_bundles(
        jobs, num_shards,
        run_family="val20_ensemble",
        profile_ids=VALIDATION20_ENSEMBLE_PROFILE_MANIFEST.keys(),
    )


VAL20_CORE_CLASSIFIER_POOL: Tuple[str, ...] = (
    "lr",
    "svm_rbf",
    "svm_linear",
    "dlda",
    "knn",
    "rf",
    "nb",
    "elastic_net_lr",
)

VAL20_EXPANDED_CLASSIFIER_POOL: Tuple[str, ...] = VAL19_HDLSS_MODERATE_NEW
VAL20_EXPANDED_CLASSIFIER_POOL_NO_TABPFN: Tuple[str, ...] = tuple(
    clf for clf in VAL20_EXPANDED_CLASSIFIER_POOL if clf != "tabpfn"
)

VAL20_CUSTOM_FLAML_FAMILIES: Tuple[str, ...] = (
    "elastic_net_lr",
    "svm_linear",
    "svm_rbf",
    "knn",
    "nb",
    "dlda",
    "shrinkage_lda",
    "nsc",
    "pls_da_classifier",
    "bc_svm_linear",
    "sglnn",
    "rff_lr",
    "copula_da",
    "cpda",
    "tabm",
    "realmlp",
    "rp_ensemble",
    "near_subspace",
)

VALIDATION20_BRIDGE_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "V20_B01_default_anchor": {
        "anchor": "yes",
        "contrast_ref": "V20_B02_ref_anchor",
        "effect": "val18_a02_bridge",
        "panel": "full64",
        "family": "B",
    },
    "V20_B02_ref_anchor": {
        "anchor": "yes",
        "contrast_ref": "V20_B01_default_anchor",
        "effect": "val18_a03_bridge",
        "panel": "full64",
        "family": "B",
    },
    "V20_B03_mnpo_ref_anchor": {
        "anchor": "yes",
        "contrast_ref": "V20_F04_sklearn_mnpo_diag24",
        "effect": "val18_c03_bridge",
        "panel": "full64",
        "family": "B",
    },
    "V20_B04_val19_new_mnpo": {
        "anchor": "yes",
        "contrast_ref": "V20_B03_mnpo_ref_anchor",
        "effect": "val19_c04_cpu_bridge",
        "panel": "full64",
        "family": "B",
    },
    "V20_B05_current_pool_mnpo": {
        "anchor": "yes",
        "contrast_ref": "V20_T01_no_tabpfn_mnpo_full64",
        "effect": "current_expanded_pool_mnpo",
        "panel": "full64",
        "family": "B",
    },
    "V20_B06_current_pool_legacy": {
        "anchor": "yes",
        "contrast_ref": "V20_T02_no_tabpfn_legacy_full64",
        "effect": "current_expanded_pool_legacy",
        "panel": "full64",
        "family": "B",
    },
}

VALIDATION20_FLAML_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "V20_F01_flaml_30s_mnpo_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_F04_sklearn_mnpo_diag24",
        "effect": "flaml_budget_30s",
        "panel": "diag24",
        "family": "F",
    },
    "V20_F02_flaml_60s_mnpo_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_F04_sklearn_mnpo_diag24",
        "effect": "flaml_budget_60s",
        "panel": "diag24",
        "family": "F",
    },
    "V20_F03_flaml_120s_mnpo_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_F04_sklearn_mnpo_diag24",
        "effect": "flaml_budget_120s",
        "panel": "diag24",
        "family": "F",
    },
    "V20_F04_sklearn_mnpo_diag24": {
        "anchor": "yes",
        "contrast_ref": "V20_F02_flaml_60s_mnpo_diag24",
        "effect": "mnpo_diag24_baseline",
        "panel": "diag24",
        "family": "F",
    },
}
for _fam in VAL20_CUSTOM_FLAML_FAMILIES:
    VALIDATION20_FLAML_PROFILE_MANIFEST[f"V20_F07_{_fam}_flaml_custom_diag24"] = {
        "anchor": "no",
        "contrast_ref": f"C_ONLY_{_fam}",
        "effect": f"flaml_custom_singleton_{_fam}",
        "panel": "diag24",
        "family": "F",
    }
del _fam

VALIDATION20_TUNE_FIRST_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "V20_TF01_tune_first_baseline_30s_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_F01_flaml_30s_mnpo_diag24",
        "effect": "tune_first_budget_30s",
        "panel": "diag24",
        "family": "TF",
    },
    "V20_TF01_tune_first_baseline_60s_diag24": {
        "anchor": "yes",
        "contrast_ref": "V20_F02_flaml_60s_mnpo_diag24",
        "effect": "tune_first_budget_60s",
        "panel": "diag24",
        "family": "TF",
    },
    "V20_TF01_tune_first_baseline_120s_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_F03_flaml_120s_mnpo_diag24",
        "effect": "tune_first_budget_120s",
        "panel": "diag24",
        "family": "TF",
    },
    "V20_TF02_tune_first_cvar_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_TF01_tune_first_baseline_60s_diag24",
        "effect": "tune_first_cvar",
        "panel": "diag24",
        "family": "TF",
    },
    "V20_TF03_tune_first_dynamic_complexity_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_TF01_tune_first_baseline_60s_diag24",
        "effect": "tune_first_dynamic_complexity",
        "panel": "diag24",
        "family": "TF",
    },
    "V20_TF04_tune_first_diversity_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_TF01_tune_first_baseline_60s_diag24",
        "effect": "tune_first_portfolio_diversity",
        "panel": "diag24",
        "family": "TF",
    },
    "V20_TF05_tune_first_full_stack_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_TF01_tune_first_baseline_60s_diag24",
        "effect": "tune_first_full_stack",
        "panel": "diag24",
        "family": "TF",
    },
    "V20_TF06_tune_first_default_plus_diversity_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_TF01_tune_first_baseline_60s_diag24",
        "effect": "tune_first_default_plus_diversity",
        "panel": "diag24",
        "family": "TF",
    },
}

VALIDATION20_TABPFN_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "V20_T01_no_tabpfn_mnpo_full64": {
        "anchor": "no",
        "contrast_ref": "V20_B05_current_pool_mnpo",
        "effect": "remove_tabpfn_mnpo",
        "panel": "full64",
        "family": "T",
    },
    "V20_T02_no_tabpfn_legacy_full64": {
        "anchor": "no",
        "contrast_ref": "V20_B06_current_pool_legacy",
        "effect": "remove_tabpfn_legacy",
        "panel": "full64",
        "family": "T",
    },
    "V20_T03_tabpfn_extreme_only_mnpo_full64": {
        "anchor": "no",
        "contrast_ref": "V20_B05_current_pool_mnpo",
        "effect": "tabpfn_extreme_only",
        "panel": "full64",
        "family": "T",
    },
}

VALIDATION20_PIPELINE_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "V20_P01_bh_alpha_010_full64": {
        "anchor": "no",
        "contrast_ref": "V20_B02_ref_anchor",
        "effect": "bh_alpha_010",
        "panel": "full64",
        "family": "P",
    },
    "V20_P02_bh_alpha_020_full64": {
        "anchor": "no",
        "contrast_ref": "V20_B02_ref_anchor",
        "effect": "bh_alpha_020",
        "panel": "full64",
        "family": "P",
    },
    "V20_P03_bh_disabled_full64": {
        "anchor": "no",
        "contrast_ref": "V20_B02_ref_anchor",
        "effect": "prefilter_disabled",
        "panel": "full64",
        "family": "P",
    },
    "V20_P04_no_gmm_after_fs_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_F04_sklearn_mnpo_diag24",
        "effect": "no_gmm_after_fs",
        "panel": "diag24",
        "family": "P",
    },
    "V20_P05_gmm_before_fs_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_F04_sklearn_mnpo_diag24",
        "effect": "gmm_before_fs",
        "panel": "diag24",
        "family": "P",
    },
    "V20_P06_no_gmm_before_fs_diag24": {
        "anchor": "no",
        "contrast_ref": "V20_P04_no_gmm_after_fs_diag24",
        "effect": "no_gmm_before_fs",
        "panel": "diag24",
        "family": "P",
    },
}

VALIDATION20_LR_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "V20_L01_lr_prior_reduced_full64": {
        "anchor": "no",
        "contrast_ref": "V20_B05_current_pool_mnpo",
        "effect": "lr_prior_reduced",
        "panel": "full64",
        "family": "L",
    },
    "V20_L02_diversity_top3_full64": {
        "anchor": "no",
        "contrast_ref": "V20_B05_current_pool_mnpo",
        "effect": "diversity_top3",
        "panel": "full64",
        "family": "L",
    },
    "V20_L03_diversity_top3_lr_prior_full64": {
        "anchor": "no",
        "contrast_ref": "V20_B05_current_pool_mnpo",
        "effect": "diversity_top3_lr_prior",
        "panel": "full64",
        "family": "L",
    },
}

VALIDATION20_WAVE1_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {}
VALIDATION20_WAVE1_PROFILE_MANIFEST.update(VALIDATION20_BRIDGE_PROFILE_MANIFEST)
VALIDATION20_WAVE1_PROFILE_MANIFEST.update(VALIDATION20_FLAML_PROFILE_MANIFEST)
VALIDATION20_WAVE1_PROFILE_MANIFEST.update(VALIDATION20_TUNE_FIRST_PROFILE_MANIFEST)
VALIDATION20_WAVE1_PROFILE_MANIFEST.update(VALIDATION20_TABPFN_PROFILE_MANIFEST)
VALIDATION20_WAVE1_PROFILE_MANIFEST.update(VALIDATION20_PIPELINE_PROFILE_MANIFEST)
VALIDATION20_WAVE1_PROFILE_MANIFEST.update(VALIDATION20_LR_PROFILE_MANIFEST)
VALIDATION20_WAVE1_PROFILE_MANIFEST.update(VALIDATION20_ENSEMBLE_PROFILE_MANIFEST)

# These follow-on profiles are intentionally documented but not emitted by the
# immediate `validation20_wave1` planner. They depend on Wave 1 evidence
# (selected FLAML budget, promoted ensemble toggles, and composite-candidate
# settings) and are carried here so docs, tooling, and future generators stay in
# sync on the reserved profile IDs.
VALIDATION20_WAVE2_RESERVED_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "V20_F05_flaml_best_mnpo_full64": {
        "anchor": "no",
        "contrast_ref": "V20_B05_current_pool_mnpo",
        "effect": "flaml_best_budget_mnpo_full64",
        "panel": "full64",
        "family": "F",
        "materialization": "post_wave1_budget_pick",
    },
    "V20_F06_flaml_best_legacy_full64": {
        "anchor": "no",
        "contrast_ref": "V20_B06_current_pool_legacy",
        "effect": "flaml_best_budget_legacy_full64",
        "panel": "full64",
        "family": "F",
        "materialization": "post_wave1_budget_pick",
    },
}

VALIDATION20_WAVE3_RESERVED_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "V20_C01_candidate_a_full64": {
        "anchor": "no",
        "contrast_ref": "V20_C04_current_default_full64",
        "effect": "candidate_a_conservative",
        "panel": "full64",
        "family": "C",
        "materialization": "post_wave2_evidence",
    },
    "V20_C02_candidate_b_full64": {
        "anchor": "no",
        "contrast_ref": "V20_C04_current_default_full64",
        "effect": "candidate_b_moderate",
        "panel": "full64",
        "family": "C",
        "materialization": "post_wave2_evidence",
    },
    "V20_C03_candidate_c_full64": {
        "anchor": "no",
        "contrast_ref": "V20_C04_current_default_full64",
        "effect": "candidate_c_aggressive",
        "panel": "full64",
        "family": "C",
        "materialization": "post_wave2_evidence",
    },
    "V20_C04_current_default_full64": {
        "anchor": "yes",
        "contrast_ref": "V20_B01_default_anchor",
        "effect": "current_default_control",
        "panel": "full64",
        "family": "C",
        "materialization": "post_wave2_evidence",
    },
}


VALIDATION20_TABARENA_W1_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "TA_W1_A_general_tabular_probe_refresh": {
        "anchor": "yes",
        "contrast_ref": "TA_W1_B_general_tabular_competitive_probe",
        "effect": "general_tabular_probe_refresh",
        "panel": "probe7",
        "benchmark_profile": "general_tabular",
        "wave": "W1",
    },
    "TA_W1_B_general_tabular_competitive_probe": {
        "anchor": "no",
        "contrast_ref": "TA_W1_A_general_tabular_probe_refresh",
        "effect": "general_tabular_competitive_probe",
        "panel": "probe7",
        "benchmark_profile": "general_tabular_competitive",
        "wave": "W1",
    },
}

VALIDATION20_TABARENA_W2_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "TA_W2_A_general_tabular_full_refresh": {
        "anchor": "yes",
        "contrast_ref": "TA_W2_B_competitive_full64",
        "effect": "general_tabular_full_refresh",
        "panel": "full38",
        "benchmark_profile": "general_tabular",
        "wave": "W2",
    },
    "TA_W2_B_competitive_full64": {
        "anchor": "yes",
        "contrast_ref": "TA_W2_A_general_tabular_full_refresh",
        "effect": "general_tabular_competitive_full38",
        "panel": "full38",
        "benchmark_profile": "general_tabular_competitive",
        "wave": "W2",
    },
    "TA_W2_C_val20_promoted_challenger": {
        "anchor": "no",
        "contrast_ref": "TA_W2_B_competitive_full64",
        "effect": "provisional_val20_promoted_challenger",
        "panel": "full38",
        "benchmark_profile": "general_tabular_competitive",
        "wave": "W2",
    },
}

VALIDATION20_TABARENA_W3_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    "TA_W3_A_general_tabular_baseline_final": {
        "anchor": "yes",
        "contrast_ref": "TA_W3_A_competitive_candidate_final",
        "effect": "final_baseline_chase",
        "panel": "full38",
        "benchmark_profile": "general_tabular",
        "wave": "W3",
    },
    "TA_W3_A_competitive_candidate_final": {
        "anchor": "yes",
        "contrast_ref": "TA_W3_A_general_tabular_baseline_final",
        "effect": "final_competitive_chase",
        "panel": "full38",
        "benchmark_profile": "general_tabular_competitive",
        "wave": "W3",
    },
    "TA_W3_A_promoted_candidate_final": {
        "anchor": "no",
        "contrast_ref": "TA_W3_A_competitive_candidate_final",
        "effect": "final_promoted_chase",
        "panel": "full38",
        "benchmark_profile": "general_tabular_competitive",
        "wave": "W3",
    },
}

VALIDATION20_TABARENA_PLAN_KINDS: Tuple[str, ...] = (
    "validation20_tabarena_w1",
    "validation20_tabarena_w2",
    "validation20_tabarena_w3",
)


def build_jobs_validation20_tabarena_w1(
    *,
    dataset_shards: int = 8,
) -> List[Job]:
    """Val-20 TabArena Wave 1: probe refresh vs competitive challenger."""
    common_job_params = {
        "dataset_panel": "probe7",
        "execution_lane": "cpu",
        "preferred_hosts": list(VAL20_TABARENA_HOSTS),
        "progress_heartbeat_sec": 30.0,
        "progress_watchdog_sec": 1800.0,
        "progress_stall_watchdog_sec": 2400.0,
        "task_timeout_sec": 7200.0,
    }
    profiles = [
        TabArenaPlanProfile(
            "TA_W1_A_general_tabular_probe_refresh",
            "general_tabular",
            dataset_sets=("general_tabular_probe",),
            seeds=(42, 52, 62),
            protocol="openml_task",
            official_fold_limit=2,
            extra_args=("--flaml-time-budget", "75"),
            skip_official_leaderboard=True,
            notes=(
                "Val-20 TabArena W1 baseline: rerun the archived general_tabular probe "
                "slice with the current core/src benchmark surface and FLAML-valid estimator list."
            ),
            job_params=common_job_params,
        ),
        TabArenaPlanProfile(
            "TA_W1_B_general_tabular_competitive_probe",
            "general_tabular_competitive",
            dataset_sets=("general_tabular_probe",),
            seeds=(42, 52, 62),
            protocol="openml_task",
            official_fold_limit=2,
            extra_args=("--flaml-time-budget", "75"),
            skip_official_leaderboard=True,
            notes=(
                "Val-20 TabArena W1 challenger: tune-first competitive profile on the same "
                "7-dataset probe slice with metric-aligned tuning and the newer ensemble controls."
            ),
            job_params=common_job_params,
        ),
    ]
    return _build_tabarena_jobs(
        run_family="val20_tabarena_w1",
        profiles=profiles,
        dataset_shards=dataset_shards,
    )


def build_jobs_validation20_tabarena_w2(
    *,
    dataset_shards: int = 8,
) -> List[Job]:
    """Val-20 TabArena Wave 2: full-catalog refresh and challengers."""
    common_job_params = {
        "dataset_panel": "full38",
        "execution_lane": "cpu",
        "preferred_hosts": list(VAL20_TABARENA_HOSTS),
        "progress_heartbeat_sec": 30.0,
        "progress_watchdog_sec": 0.0,
        "progress_stall_watchdog_sec": 1800.0,
        "task_timeout_sec": 14400.0,
    }
    profiles = [
        TabArenaPlanProfile(
            "TA_W2_A_general_tabular_full_refresh",
            "general_tabular",
            dataset_sets=("all",),
            seeds=(42,),
            protocol="openml_task",
            official_fold_limit=0,
            skip_official_leaderboard=False,
            leaderboard_method_name="tabnetics_general_tabular_refresh",
            notes="Val-20 TabArena W2 baseline: full official rerun of general_tabular on all classification datasets.",
            job_params=common_job_params,
        ),
        TabArenaPlanProfile(
            "TA_W2_B_competitive_full64",
            "general_tabular_competitive",
            dataset_sets=("all",),
            seeds=(42,),
            protocol="openml_task",
            official_fold_limit=0,
            skip_official_leaderboard=False,
            leaderboard_method_name="tabnetics_general_tabular_competitive",
            notes="Val-20 TabArena W2 challenger: full official rerun of general_tabular_competitive.",
            job_params=common_job_params,
        ),
        TabArenaPlanProfile(
            "TA_W2_C_val20_promoted_challenger",
            "general_tabular_competitive",
            dataset_sets=("all",),
            seeds=(42,),
            protocol="openml_task",
            official_fold_limit=0,
            extra_args=("--enable-classifier-oracle-cvar",),
            skip_official_leaderboard=False,
            leaderboard_method_name="tabnetics_general_tabular_promoted",
            notes=(
                "Val-20 TabArena W2 provisional promoted challenger: general_tabular_competitive "
                "plus the most likely next classifier-oracle promotion knob (CVaR) until Wave 1 evidence "
                "selects the final promoted stack."
            ),
            job_params=common_job_params,
        ),
    ]
    return _build_tabarena_jobs(
        run_family="val20_tabarena_w2",
        profiles=profiles,
        dataset_shards=dataset_shards,
    )


def build_jobs_validation20_tabarena_w3(
    *,
    dataset_shards: int = 8,
) -> List[Job]:
    """Val-20 TabArena Wave 3: final chase between the top candidates and baseline."""
    common_job_params = {
        "dataset_panel": "full38",
        "execution_lane": "cpu",
        "preferred_hosts": list(VAL20_TABARENA_HOSTS),
        "progress_heartbeat_sec": 30.0,
        "progress_watchdog_sec": 0.0,
        "progress_stall_watchdog_sec": 1800.0,
        "task_timeout_sec": 14400.0,
    }
    profiles = [
        TabArenaPlanProfile(
            "TA_W3_A_general_tabular_baseline_final",
            "general_tabular",
            dataset_sets=("all",),
            seeds=(42,),
            protocol="openml_task",
            official_fold_limit=0,
            skip_official_leaderboard=False,
            leaderboard_method_name="tabnetics_general_tabular_final_baseline",
            notes="Val-20 TabArena W3 final chase baseline: operational general_tabular rerun with final official-fold artifacts.",
            job_params=common_job_params,
        ),
        TabArenaPlanProfile(
            "TA_W3_A_competitive_candidate_final",
            "general_tabular_competitive",
            dataset_sets=("all",),
            seeds=(42,),
            protocol="openml_task",
            official_fold_limit=0,
            skip_official_leaderboard=False,
            leaderboard_method_name="tabnetics_general_tabular_final_competitive",
            notes="Val-20 TabArena W3 final chase candidate: competitive tune-first stack with final official-fold artifacts.",
            job_params=common_job_params,
        ),
        TabArenaPlanProfile(
            "TA_W3_A_promoted_candidate_final",
            "general_tabular_competitive",
            dataset_sets=("all",),
            seeds=(42,),
            protocol="openml_task",
            official_fold_limit=0,
            extra_args=("--enable-classifier-oracle-cvar",),
            skip_official_leaderboard=False,
            leaderboard_method_name="tabnetics_general_tabular_final_promoted",
            notes=(
                "Val-20 TabArena W3 final chase promoted candidate: current best provisional "
                "Val-20 promotion stack for general tabular, ready to be regenerated after Wave 2 if needed."
            ),
            job_params=common_job_params,
        ),
    ]
    return _build_tabarena_jobs(
        run_family="val20_tabarena_w3",
        profiles=profiles,
        dataset_shards=dataset_shards,
    )


def _balanced_shard_assign_validation20_tabarena_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-20 TabArena wave jobs by task-shard bundle for aligned profile comparisons."""
    grouped: Dict[Tuple[str, str], List[Job]] = {}
    ungrouped: List[Job] = []
    for job in jobs:
        parts = str(job.job_id).split("/")
        if len(parts) != 3 or not str(parts[0]).startswith("val20_tabarena_"):
            ungrouped.append(job)
            continue
        panel = str(dict(job.params or {}).get("dataset_panel", "") or "").strip().lower() or "full38"
        grouped.setdefault((panel, str(parts[2])), []).append(job)

    bundles: List[Tuple[float, List[Job]]] = []
    for bundle_jobs in grouped.values():
        bundle_weight = sum(float(j.weight) for j in bundle_jobs)
        bundles.append((bundle_weight, sorted(bundle_jobs, key=lambda j: j.job_id)))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for bundle_weight, bundle_jobs in sorted(bundles, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        for bundle_job in bundle_jobs:
            shards[target].append(bundle_job.job_id)
        totals[target] += float(bundle_weight)

    for job in sorted(ungrouped, key=lambda j: float(j.weight), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        shards[target].append(job.job_id)
        totals[target] += float(job.weight)

    return shards


def build_jobs_validation20_wave1(
    *,
    dataset_shards: int = 8,
    val17_root: Optional[Path] = None,
) -> List[Job]:
    """Val-20 Wave 1: the currently runnable main-validation Val-20 lane.

    Evidence-dependent follow-on profiles (F05/F06 and C01-C04) are documented
    in `VALIDATION20_WAVE2_RESERVED_PROFILE_MANIFEST` and
    `VALIDATION20_WAVE3_RESERVED_PROFILE_MANIFEST`, but are intentionally not
    emitted here until Wave 1 picks the concrete budgets/promotions.
    """
    benchmark_datasets, _ = _load_benchmark_registry()
    if val17_root is None:
        val17_root = REPO_ROOT / "run_artifacts" / "validation-17"
    val17_runtime = _load_runtime_hints_from_summaries(val17_root, phase_tag="val17")
    diag24_ids = list(VAL18_DIAG24)
    full64_ids = list(VAL18_FULL64)
    runtime_est_diag24 = _val18_runtime_estimate(diag24_ids, benchmark_datasets, val17_runtime)
    runtime_est_full64 = _val18_runtime_estimate(full64_ids, benchmark_datasets, val17_runtime)

    seeds = [11, 23, 37, 42, 59]
    ref = _val18_ref_flags()
    fs_method_set = "mnpo_v14_core_plus_ipss"

    cpu_diag_job_params = {
        "dataset_panel": "diag24",
        "execution_lane": "cpu",
        "preferred_hosts": list(VAL20_CPU_HOSTS),
    }
    cpu_full64_job_params = {
        "dataset_panel": "full64",
        "execution_lane": "cpu",
        "preferred_hosts": list(VAL20_CPU_HOSTS),
    }
    gpu_full64_job_params = {
        "dataset_panel": "full64",
        "execution_lane": "tabpfn",
        "preferred_hosts": list(VAL20_GPU_HOSTS),
    }

    core_pool = tuple(VAL20_CORE_CLASSIFIER_POOL)
    expanded_pool = tuple(VAL20_EXPANDED_CLASSIFIER_POOL)
    expanded_pool_no_tabpfn = tuple(VAL20_EXPANDED_CLASSIFIER_POOL_NO_TABPFN)
    core_pool_max = str(len(core_pool))
    expanded_pool_max = str(len(expanded_pool))
    expanded_pool_no_tabpfn_max = str(len(expanded_pool_no_tabpfn))

    legacy_core_base: Tuple[str, ...] = (
        *ref,
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates", *core_pool,
        "--model-cv-runtime-max-candidates", core_pool_max,
    )
    mnpo_core_base: Tuple[str, ...] = (
        *ref,
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classification-backend", "sklearn",
        "--classifier-oracle-behavior-profile", "current",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--model-candidates", *core_pool,
        "--model-cv-runtime-max-candidates", core_pool_max,
    )
    legacy_expanded_base: Tuple[str, ...] = (
        *ref,
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
        "--model-candidates", *expanded_pool,
        "--model-cv-runtime-max-candidates", expanded_pool_max,
    )
    mnpo_expanded_base: Tuple[str, ...] = (
        *ref,
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classification-backend", "sklearn",
        "--classifier-oracle-behavior-profile", "current",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--model-candidates", *expanded_pool,
        "--model-cv-runtime-max-candidates", expanded_pool_max,
    )
    mnpo_expanded_cpu_bridge: Tuple[str, ...] = (
        *ref,
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classification-backend", "sklearn",
        "--classifier-oracle-behavior-profile", "current",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--model-candidates", *VAL19_HDLSS_MODERATE_CPU_NEW,
        "--model-cv-runtime-max-candidates", str(len(VAL19_HDLSS_MODERATE_CPU_NEW)),
    )
    mnpo_flaml_core_base: Tuple[str, ...] = (
        *ref,
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classification-backend", "flaml",
        "--classifier-oracle-behavior-profile", "current",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--model-candidates", *core_pool,
        "--model-cv-runtime-max-candidates", core_pool_max,
    )
    tune_first_core_base: Tuple[str, ...] = (
        *ref,
        "--classifier-selection-mode", "tune_first",
        "--classification-backend", "flaml",
        "--classifier-oracle-behavior-profile", "current",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--model-candidates", *core_pool,
        "--model-cv-runtime-max-candidates", core_pool_max,
    )

    bridge_profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            "V20_B01_default_anchor",
            "mnpo_broad_all",
            _val18_d_default_flags(),
            notes="Val-20 bridge: exact Val-18 A02 rerun on overlapping seeds.",
            job_params=cpu_full64_job_params,
        ),
        BenchmarkProfile(
            "V20_B02_ref_anchor",
            fs_method_set,
            _val18_ref_flags(),
            notes="Val-20 bridge: exact Val-18 A03 rerun on overlapping seeds.",
            job_params=cpu_full64_job_params,
        ),
        BenchmarkProfile(
            "V20_B03_mnpo_ref_anchor",
            fs_method_set,
            mnpo_core_base,
            notes="Val-20 bridge: closest Val-18 MNPO reference (C03 core pool) on FULL64.",
            job_params=cpu_full64_job_params,
        ),
        BenchmarkProfile(
            "V20_B04_val19_new_mnpo",
            fs_method_set,
            mnpo_expanded_cpu_bridge,
            notes="Val-20 bridge: exact Val-19 V19_C04 CPU-stable pool rerun.",
            job_params={**cpu_full64_job_params, "pool_snapshot_id": "val19_cpu_new"},
        ),
        BenchmarkProfile(
            "V20_B05_current_pool_mnpo",
            fs_method_set,
            mnpo_expanded_base,
            weight_mult=1.4,
            notes="Val-20 operational baseline: expanded current MNPO pool including TabPFN.",
            job_params={**gpu_full64_job_params, "pool_snapshot_id": "current_expanded"},
        ),
        BenchmarkProfile(
            "V20_B06_current_pool_legacy",
            fs_method_set,
            legacy_expanded_base,
            weight_mult=1.4,
            notes="Val-20 operational baseline: expanded current legacy pool including TabPFN.",
            job_params={**gpu_full64_job_params, "pool_snapshot_id": "current_expanded"},
        ),
    ]

    flaml_profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            "V20_F01_flaml_30s_mnpo_diag24",
            fs_method_set,
            (*mnpo_flaml_core_base, "--flaml-time-budget", "30"),
            weight_mult=1.4,
            notes="Val-20 F1: MNPO + per-family FLAML (30s budget).",
            job_params=cpu_diag_job_params,
        ),
        BenchmarkProfile(
            "V20_F02_flaml_60s_mnpo_diag24",
            fs_method_set,
            (*mnpo_flaml_core_base, "--flaml-time-budget", "60"),
            weight_mult=1.7,
            notes="Val-20 F1: MNPO + per-family FLAML (60s budget).",
            job_params=cpu_diag_job_params,
        ),
        BenchmarkProfile(
            "V20_F03_flaml_120s_mnpo_diag24",
            fs_method_set,
            (*mnpo_flaml_core_base, "--flaml-time-budget", "120"),
            weight_mult=2.0,
            notes="Val-20 F1: MNPO + per-family FLAML (120s budget).",
            job_params=cpu_diag_job_params,
        ),
        BenchmarkProfile(
            "V20_F04_sklearn_mnpo_diag24",
            fs_method_set,
            mnpo_core_base,
            notes="Val-20 F1 baseline: MNPO without FLAML on the same DIAG24 panel.",
            job_params=cpu_diag_job_params,
        ),
    ]
    for family_name in VAL20_CUSTOM_FLAML_FAMILIES:
        flaml_profiles.append(
            BenchmarkProfile(
                f"V20_F07_{family_name}_flaml_custom_diag24",
                fs_method_set,
                (
                    *ref,
                    "--classifier-selection-mode", "mnpo_hybrid",
                    "--classification-backend", "flaml",
                    "--classifier-oracle-behavior-profile", "current",
                    "--classifier-oracle-k", "1",
                    "--model-candidates", family_name,
                    "--model-cv-runtime-max-candidates", "1",
                    "--flaml-time-budget", "60",
                ),
                weight_mult=1.6,
                notes=f"Val-20 F3: singleton FLAML custom-learner diagnostic for {family_name}.",
                job_params=cpu_diag_job_params,
            )
        )

    tune_first_profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            "V20_TF01_tune_first_baseline_30s_diag24",
            fs_method_set,
            (*tune_first_core_base, "--flaml-time-budget", "30"),
            weight_mult=2.0,
            notes="Val-20 TF baseline: tune-first with 30s budget.",
            job_params=cpu_diag_job_params,
        ),
        BenchmarkProfile(
            "V20_TF01_tune_first_baseline_60s_diag24",
            fs_method_set,
            (*tune_first_core_base, "--flaml-time-budget", "60"),
            weight_mult=2.3,
            notes="Val-20 TF baseline: tune-first with 60s budget.",
            job_params=cpu_diag_job_params,
        ),
        BenchmarkProfile(
            "V20_TF01_tune_first_baseline_120s_diag24",
            fs_method_set,
            (*tune_first_core_base, "--flaml-time-budget", "120"),
            weight_mult=2.6,
            notes="Val-20 TF baseline: tune-first with 120s budget.",
            job_params=cpu_diag_job_params,
        ),
        BenchmarkProfile(
            "V20_TF02_tune_first_cvar_diag24",
            fs_method_set,
            (*tune_first_core_base, "--flaml-time-budget", "60", "--enable-classifier-oracle-cvar"),
            weight_mult=2.3,
            notes="Val-20 TF: add CVaR to the classifier oracle.",
            job_params=cpu_diag_job_params,
        ),
        BenchmarkProfile(
            "V20_TF03_tune_first_dynamic_complexity_diag24",
            fs_method_set,
            (*tune_first_core_base, "--flaml-time-budget", "60", "--enable-classifier-oracle-dynamic-complexity"),
            weight_mult=2.3,
            notes="Val-20 TF: dynamic complexity override in tune-first mode.",
            job_params=cpu_diag_job_params,
        ),
        BenchmarkProfile(
            "V20_TF04_tune_first_diversity_diag24",
            fs_method_set,
            (*tune_first_core_base, "--flaml-time-budget", "60", "--enable-classifier-oracle-portfolio-diversity"),
            weight_mult=2.3,
            notes="Val-20 TF: portfolio diversity extraction in tune-first mode.",
            job_params=cpu_diag_job_params,
        ),
        BenchmarkProfile(
            "V20_TF05_tune_first_full_stack_diag24",
            fs_method_set,
            (
                *tune_first_core_base,
                "--flaml-time-budget", "60",
                "--enable-classifier-oracle-cvar",
                "--enable-classifier-oracle-dynamic-complexity",
                "--enable-classifier-oracle-portfolio-diversity",
            ),
            weight_mult=2.4,
            notes="Val-20 TF: CVaR + dynamic complexity + portfolio diversity.",
            job_params=cpu_diag_job_params,
        ),
        BenchmarkProfile(
            "V20_TF06_tune_first_default_plus_diversity_diag24",
            fs_method_set,
            (
                *tune_first_core_base,
                "--flaml-time-budget", "60",
                "--enable-classifier-oracle-portfolio-diversity",
                "--classifier-oracle-k", "3",
            ),
            weight_mult=2.4,
            notes="Val-20 TF: default tune-first oracle with diversified top-3 extraction.",
            job_params=cpu_diag_job_params,
        ),
    ]

    tabpfn_profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            "V20_T01_no_tabpfn_mnpo_full64",
            fs_method_set,
            (
                *ref,
                "--classifier-selection-mode", "mnpo_hybrid",
                "--classification-backend", "sklearn",
                "--classifier-oracle-behavior-profile", "current",
                "--classifier-oracle-k", "2",
                "--classifier-oracle-weighting-mode", "tritrust",
                "--classifier-oracle-bbc-bootstrap-rounds", "120",
                "--classifier-oracle-bbc-ci-level", "0.90",
                "--enable-classifier-oracle-ensemble",
                "--model-candidates", *expanded_pool_no_tabpfn,
                "--model-cv-runtime-max-candidates", expanded_pool_no_tabpfn_max,
            ),
            notes="Val-20 T: remove TabPFN from the expanded MNPO pool.",
            job_params=cpu_full64_job_params,
        ),
        BenchmarkProfile(
            "V20_T02_no_tabpfn_legacy_full64",
            fs_method_set,
            (
                *ref,
                "--classifier-selection-mode", "legacy",
                "--classification-backend", "sklearn",
                "--model-candidates", *expanded_pool_no_tabpfn,
                "--model-cv-runtime-max-candidates", expanded_pool_no_tabpfn_max,
            ),
            notes="Val-20 T: remove TabPFN from the expanded legacy pool.",
            job_params=cpu_full64_job_params,
        ),
        BenchmarkProfile(
            "V20_T03_tabpfn_extreme_only_mnpo_full64",
            fs_method_set,
            (
                *mnpo_expanded_base,
                "--classifier-regime-candidate-exclusions", "hdlss_moderate:tabpfn", "standard:tabpfn",
            ),
            weight_mult=1.5,
            notes="Val-20 T: keep TabPFN only in the extreme HDLSS regime.",
            job_params=gpu_full64_job_params,
        ),
    ]

    pipeline_full64_profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            "V20_P01_bh_alpha_010_full64",
            fs_method_set,
            (*_val18_ref_flags(), "--prefilter-bh-ttest-alpha", "0.10"),
            notes="Val-20 P1: relaxed BH FDR alpha at 0.10.",
            job_params=cpu_full64_job_params,
        ),
        BenchmarkProfile(
            "V20_P02_bh_alpha_020_full64",
            fs_method_set,
            (*_val18_ref_flags(), "--prefilter-bh-ttest-alpha", "0.20"),
            notes="Val-20 P1: relaxed BH FDR alpha at 0.20.",
            job_params=cpu_full64_job_params,
        ),
        BenchmarkProfile(
            "V20_P03_bh_disabled_full64",
            fs_method_set,
            (*_val18_ref_flags(), "--disable-prefilter", "--no-prefilter-bh-ttest"),
            notes="Val-20 P1: disable the prefilter stage entirely.",
            job_params=cpu_full64_job_params,
        ),
    ]

    # Phase P2: 2×2 factorial — multimodal fallback (gmm vs none) × stage ordering
    # (after_fs vs before_fs).  The reference B01 already uses gmm + after_fs, so
    # P04 disables GMM fallback, P05 reverses ordering, and P06 combines both.
    # NOTE: The original implementation incorrectly used --dist-criterion gmm
    # (not a valid argparse choice).  GMM multimodal detection is controlled by
    # --df-multimodal-fallback, which defaults to "gmm".  Fixed 2026-03-15.
    no_gmm_after_fs_flags = _replace_flag_value(
        _val18_ref_flags(), "--df-multimodal-fallback", "none"
    )
    gmm_before_fs_flags = _replace_flag_value(
        _val18_ref_flags(), "--df-stage-position", "before_fs"
    )
    no_gmm_before_fs_flags = _replace_flag_value(
        no_gmm_after_fs_flags, "--df-stage-position", "before_fs"
    )
    pipeline_diag_profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            "V20_P04_no_gmm_after_fs_diag24",
            fs_method_set,
            no_gmm_after_fs_flags,
            notes="Val-20 P2: multimodal fallback disabled, after-FS ordering (control vs ref B01 gmm+after_fs).",
            job_params=cpu_diag_job_params,
        ),
        BenchmarkProfile(
            "V20_P05_gmm_before_fs_diag24",
            fs_method_set,
            gmm_before_fs_flags,
            notes="Val-20 P2: GMM multimodal fallback (default) with before-FS ordering.",
            job_params=cpu_diag_job_params,
        ),
        BenchmarkProfile(
            "V20_P06_no_gmm_before_fs_diag24",
            fs_method_set,
            no_gmm_before_fs_flags,
            notes="Val-20 P2: multimodal fallback disabled with before-FS ordering.",
            job_params=cpu_diag_job_params,
        ),
    ]

    lr_profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            "V20_L01_lr_prior_reduced_full64",
            fs_method_set,
            (*mnpo_expanded_base, "--classifier-complexity-prior-override", "lr=0.75"),
            weight_mult=1.5,
            notes="Val-20 L: reduce the LR complexity prior under the expanded MNPO pool.",
            job_params=gpu_full64_job_params,
        ),
        BenchmarkProfile(
            "V20_L02_diversity_top3_full64",
            fs_method_set,
            (
                *mnpo_expanded_base,
                "--classifier-oracle-k", "3",
                "--enable-classifier-oracle-portfolio-diversity",
                "--enable-classifier-oracle-ensemble",
            ),
            weight_mult=1.6,
            notes="Val-20 L: diversify the expanded MNPO pool with top-3 extraction.",
            job_params=gpu_full64_job_params,
        ),
        BenchmarkProfile(
            "V20_L03_diversity_top3_lr_prior_full64",
            fs_method_set,
            (
                *mnpo_expanded_base,
                "--classifier-oracle-k", "3",
                "--enable-classifier-oracle-portfolio-diversity",
                "--enable-classifier-oracle-ensemble",
                "--classifier-complexity-prior-override", "lr=0.75",
            ),
            weight_mult=1.6,
            notes="Val-20 L: top-3 diversity extraction plus reduced LR prior.",
            job_params=gpu_full64_job_params,
        ),
    ]

    all_profiles = [
        *bridge_profiles,
        *flaml_profiles,
        *tune_first_profiles,
        *tabpfn_profiles,
        *pipeline_full64_profiles,
        *pipeline_diag_profiles,
        *lr_profiles,
    ]
    _val18_validate_method_sets(all_profiles, "validation20_wave1")

    jobs: List[Job] = []
    jobs.extend(
        _val18_build_jobs(
            run_family="val20_bridge",
            profiles=bridge_profiles,
            dataset_ids=full64_ids,
            seeds=seeds,
            dataset_shards=dataset_shards,
            runtime_est=runtime_est_full64,
        )
    )
    jobs.extend(
        _val18_build_jobs(
            run_family="val20_flaml",
            profiles=flaml_profiles,
            dataset_ids=diag24_ids,
            seeds=seeds,
            dataset_shards=dataset_shards,
            runtime_est=runtime_est_diag24,
        )
    )
    jobs.extend(
        _val18_build_jobs(
            run_family="val20_tune_first",
            profiles=tune_first_profiles,
            dataset_ids=diag24_ids,
            seeds=seeds,
            dataset_shards=dataset_shards,
            runtime_est=runtime_est_diag24,
        )
    )
    jobs.extend(
        _val18_build_jobs(
            run_family="val20_tabpfn",
            profiles=tabpfn_profiles,
            dataset_ids=full64_ids,
            seeds=seeds,
            dataset_shards=dataset_shards,
            runtime_est=runtime_est_full64,
        )
    )
    jobs.extend(
        _val18_build_jobs(
            run_family="val20_pipeline",
            profiles=pipeline_full64_profiles,
            dataset_ids=full64_ids,
            seeds=seeds,
            dataset_shards=dataset_shards,
            runtime_est=runtime_est_full64,
        )
    )
    jobs.extend(
        _val18_build_jobs(
            run_family="val20_pipeline",
            profiles=pipeline_diag_profiles,
            dataset_ids=diag24_ids,
            seeds=seeds,
            dataset_shards=dataset_shards,
            runtime_est=runtime_est_diag24,
        )
    )
    jobs.extend(
        _val18_build_jobs(
            run_family="val20_lr",
            profiles=lr_profiles,
            dataset_ids=full64_ids,
            seeds=seeds,
            dataset_shards=dataset_shards,
            runtime_est=runtime_est_full64,
        )
    )
    jobs.extend(
        build_jobs_validation20_ensemble(
            dataset_shards=dataset_shards,
            val17_root=val17_root,
        )
    )
    return jobs


def _balanced_shard_assign_validation20_wave1_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-20 Wave 1 by lane/panel dataset bundles for aligned comparisons."""
    grouped: Dict[Tuple[str, str, str], List[Job]] = {}
    ungrouped: List[Job] = []
    for job in jobs:
        parts = str(job.job_id).split("/")
        if len(parts) != 3 or not str(parts[0]).startswith("val20_"):
            ungrouped.append(job)
            continue
        params = dict(job.params or {})
        lane = str(params.get("execution_lane", "") or "").strip().lower() or "cpu"
        panel = str(params.get("dataset_panel", "") or "").strip().lower() or "diag24"
        grouped.setdefault((lane, panel, str(parts[2])), []).append(job)

    cpu_bundles: List[Tuple[float, List[Job]]] = []
    gpu_bundles: List[Tuple[float, List[Job]]] = []
    for (lane, _panel, _part), bundle_jobs in grouped.items():
        bundle_weight = sum(float(j.weight) for j in bundle_jobs)
        record = (bundle_weight, sorted(bundle_jobs, key=lambda j: j.job_id))
        if lane == "tabpfn":
            gpu_bundles.append(record)
        else:
            cpu_bundles.append(record)

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    cpu_weight = sum(float(w) for w, _ in cpu_bundles)
    gpu_weight = sum(float(w) for w, _ in gpu_bundles)
    if cpu_bundles and gpu_bundles:
        total_weight = float(cpu_weight + gpu_weight)
        gpu_shards = int(round(float(num_shards) * float(gpu_weight / max(total_weight, 1.0))))
        gpu_shards = max(1, min(int(num_shards) - 1, gpu_shards))
        cpu_shards = int(num_shards) - gpu_shards
        cpu_shard_ids = list(range(1, cpu_shards + 1))
        gpu_shard_ids = list(range(cpu_shards + 1, int(num_shards) + 1))
    elif gpu_bundles:
        cpu_shard_ids = []
        gpu_shard_ids = list(range(1, int(num_shards) + 1))
    else:
        cpu_shard_ids = list(range(1, int(num_shards) + 1))
        gpu_shard_ids = []

    def _assign(bundle_items: Sequence[Tuple[float, List[Job]]], shard_ids: Sequence[int]) -> None:
        if not shard_ids:
            return
        for bundle_weight, bundle_jobs in sorted(bundle_items, key=lambda x: float(x[0]), reverse=True):
            target = min(((sid, totals[sid]) for sid in shard_ids), key=lambda kv: kv[1])[0]
            for bundle_job in bundle_jobs:
                shards[target].append(bundle_job.job_id)
            totals[target] += float(bundle_weight)

    _assign(cpu_bundles, cpu_shard_ids)
    _assign(gpu_bundles, gpu_shard_ids)

    fallback_ids = cpu_shard_ids or gpu_shard_ids or list(range(1, int(num_shards) + 1))
    for job in sorted(ungrouped, key=lambda j: float(j.weight), reverse=True):
        target = min(((sid, totals[sid]) for sid in fallback_ids), key=lambda kv: kv[1])[0]
        shards[target].append(job.job_id)
        totals[target] += float(job.weight)

    return shards


# ---- Family W: Classifier Oracle Weighting Matrix ----

VALIDATION18_CLS_ORACLE_WT_PROFILE_MANIFEST: Dict[str, Dict[str, str]] = {
    # Tier 1: core weighting sweep (8-clf legacy pool).
    "W01_cls_oracle_tritrust": {"anchor": "yes", "contrast_ref": "W03_cls_oracle_banzhaf", "effect": "cls_oracle_tritrust"},
    "W02_cls_oracle_uniform": {"anchor": "no", "contrast_ref": "W01_cls_oracle_tritrust", "effect": "cls_oracle_uniform"},
    "W03_cls_oracle_banzhaf": {"anchor": "yes", "contrast_ref": "W01_cls_oracle_tritrust", "effect": "cls_oracle_banzhaf"},
    "W04_cls_oracle_shapley": {"anchor": "no", "contrast_ref": "W03_cls_oracle_banzhaf", "effect": "cls_oracle_shapley"},
    # Tier 2: cross-interaction with FS oracle weighting.
    "W05_cross_fs_tritrust_cls_banzhaf": {"anchor": "no", "contrast_ref": "W03_cls_oracle_banzhaf", "effect": "cross_fs_tritrust_cls_banzhaf"},
    "W06_cross_fs_uniform_cls_banzhaf": {"anchor": "no", "contrast_ref": "W03_cls_oracle_banzhaf", "effect": "cross_fs_uniform_cls_banzhaf"},
    "W07_cross_fs_shapley_cls_banzhaf": {"anchor": "no", "contrast_ref": "W03_cls_oracle_banzhaf", "effect": "cross_fs_shapley_cls_banzhaf"},
    # Tier 3: pool-size interaction.
    "W08_broad_pool_tritrust": {"anchor": "no", "contrast_ref": "W01_cls_oracle_tritrust", "effect": "broad_pool_tritrust"},
    "W09_broad_pool_banzhaf": {"anchor": "no", "contrast_ref": "W03_cls_oracle_banzhaf", "effect": "broad_pool_banzhaf"},
}


def build_jobs_validation18_cls_oracle_wt(
    *,
    dataset_shards: int = 7,
    val17_root: Optional[Path] = None,
) -> List[Job]:
    """Val-18 Family W: classifier-oracle weighting sweep on DIAG24."""
    benchmark_datasets, _ = _load_benchmark_registry()
    if val17_root is None:
        val17_root = REPO_ROOT / "run_artifacts" / "validation-17"
    val17_runtime = _load_runtime_hints_from_summaries(val17_root, phase_tag="val17")
    diag24_ids = list(VAL18_DIAG24)
    runtime_est = _val18_runtime_estimate(diag24_ids, benchmark_datasets, val17_runtime)

    seeds = [11, 23, 37, 42, 59]  # 5 seeds
    ref = _val18_ref_flags()
    fs_method_set = "mnpo_v14_core_plus_ipss"

    legacy_pool_models: Tuple[str, ...] = (
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
    )
    broad_pool_models: Tuple[str, ...] = (
        *legacy_pool_models, "shrinkage_lda", "nsc", "pls_da_classifier", "gpc",
    )

    cpu_job_params = {
        "dataset_panel": "diag24",
        "execution_lane": "cpu",
        "preferred_hosts": list(VAL18_CPU_HOSTS),
    }

    # Base MNPO hybrid classifier flags (shared across W-profiles).
    def _mnpo_clf_flags(
        weighting_mode: str,
        model_candidates: Tuple[str, ...],
        *,
        fs_oracle_override: Optional[str] = None,
    ) -> Tuple[str, ...]:
        base = ref if fs_oracle_override is None else _replace_flag_value(ref, "--fs-oracle-weighting-mode", fs_oracle_override)
        return (
            *base,
            "--classifier-selection-mode", "mnpo_hybrid",
            "--classifier-oracle-k", "2",
            "--classifier-oracle-weighting-mode", weighting_mode,
            "--classifier-oracle-bbc-bootstrap-rounds", "120",
            "--classifier-oracle-bbc-ci-level", "0.90",
            "--enable-classifier-oracle-ensemble",
            "--model-candidates", *model_candidates,
            "--model-cv-runtime-max-candidates", str(len(model_candidates)),
        )

    profiles: List[BenchmarkProfile] = [
        # Tier 1: core weighting sweep.
        BenchmarkProfile("W01_cls_oracle_tritrust", fs_method_set,
                         _mnpo_clf_flags("tritrust", legacy_pool_models),
                         notes="Val-18 W: classifier oracle tritrust (current default).",
                         job_params=cpu_job_params),
        BenchmarkProfile("W02_cls_oracle_uniform", fs_method_set,
                         _mnpo_clf_flags("uniform", legacy_pool_models),
                         notes="Val-18 W: classifier oracle uniform (no-smart-weighting control).",
                         job_params=cpu_job_params),
        BenchmarkProfile("W03_cls_oracle_banzhaf", fs_method_set,
                         _mnpo_clf_flags("banzhaf", legacy_pool_models),
                         notes="Val-18 W: classifier oracle banzhaf (primary hypothesis).",
                         job_params=cpu_job_params),
        BenchmarkProfile("W04_cls_oracle_shapley", fs_method_set,
                         _mnpo_clf_flags("shapley", legacy_pool_models),
                         notes="Val-18 W: classifier oracle shapley.",
                         job_params=cpu_job_params),
        # Tier 2: cross-interaction with FS oracle weighting.
        BenchmarkProfile("W05_cross_fs_tritrust_cls_banzhaf", fs_method_set,
                         _mnpo_clf_flags("banzhaf", legacy_pool_models, fs_oracle_override="tritrust"),
                         notes="Val-18 W: FS-oracle tritrust × classifier-oracle banzhaf.",
                         job_params=cpu_job_params),
        BenchmarkProfile("W06_cross_fs_uniform_cls_banzhaf", fs_method_set,
                         _mnpo_clf_flags("banzhaf", legacy_pool_models, fs_oracle_override="uniform"),
                         notes="Val-18 W: FS-oracle uniform × classifier-oracle banzhaf.",
                         job_params=cpu_job_params),
        BenchmarkProfile("W07_cross_fs_shapley_cls_banzhaf", fs_method_set,
                         _mnpo_clf_flags("banzhaf", legacy_pool_models, fs_oracle_override="shapley"),
                         notes="Val-18 W: FS-oracle shapley × classifier-oracle banzhaf.",
                         job_params=cpu_job_params),
        # Tier 3: pool-size interaction.
        BenchmarkProfile("W08_broad_pool_tritrust", fs_method_set,
                         _mnpo_clf_flags("tritrust", broad_pool_models),
                         notes="Val-18 W: 12-clf pool × tritrust (pool-size interaction).",
                         job_params=cpu_job_params),
        BenchmarkProfile("W09_broad_pool_banzhaf", fs_method_set,
                         _mnpo_clf_flags("banzhaf", broad_pool_models),
                         notes="Val-18 W: 12-clf pool × banzhaf (pool-size interaction).",
                         job_params=cpu_job_params),
    ]

    _val18_validate_method_sets(profiles, "validation18_cls_oracle_wt")

    return _val18_build_jobs(
        run_family="val18_cls_oracle_wt",
        profiles=profiles,
        dataset_ids=diag24_ids,
        seeds=seeds,
        dataset_shards=dataset_shards,
        runtime_est=runtime_est,
    )


def _balanced_shard_assign_validation18_cls_oracle_wt_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """CPU-only shard assignment for Family W."""
    return _balanced_shard_assign_profile_bundles(
        jobs, num_shards,
        run_family="val18_cls_oracle_wt",
        profile_ids=VALIDATION18_CLS_ORACLE_WT_PROFILE_MANIFEST.keys(),
    )


# ---------------------------------------------------------------------------
# Validation-13 classifier oracle comparison (standalone supplement)
# ---------------------------------------------------------------------------


def build_jobs_validation13_clf_oracle(
    *,
    dataset_shards: int = 4,
    val12_root: Optional[Path] = None,
) -> List[Job]:
    """Validation-13 classifier oracle comparison — standalone 2×2 factorial.

    Isolates the classifier selection algorithm effect from the candidate pool
    size by running a clean 2×2 design on the same MNPO FS base:

    +-----------------+--------------------+---------------------+
    |                 | 8-clf pool         | 16-clf pool         |
    +-----------------+--------------------+---------------------+
    | Legacy CV       | clf_legacy_pool8   | clf_legacy_pool16   |
    | MNPO hybrid     | clf_mnpo_pool8     | clf_mnpo_pool16     |
    +-----------------+--------------------+---------------------+

    All four profiles share the same MNPO FS path (d_default style: banzhaf
    oracle weighting, evalue screening, copula derandomize, etc.) so only
    the classifier stage varies.  Uses the same 35 Val-13 datasets and 9 seeds.
    Results can be analysed independently or merged with Val-13 main results.

    Profiles (4):
      - clf_legacy_pool8:   MNPO FS + legacy CV clf selection, 8-candidate pool
      - clf_legacy_pool16:  MNPO FS + legacy CV clf selection, 16-candidate pool
      - clf_mnpo_pool8:     MNPO FS + MNPO hybrid clf selection, 8-candidate pool
      - clf_mnpo_pool16:    MNPO FS + MNPO hybrid clf selection, 16-candidate pool
    """
    VAL13_DATASETS: List[str] = [
        # Val-12 core (30)
        "arcene_nips03",
        "carcinom_11class",
        "cll_sub_111",
        "cns_pomeroy",
        "colon_alon",
        "cumida_brain_gse50161",
        "cumida_colorectal_gse44861",
        "cumida_leukemia_subtypes",
        "dlbcl_shipp",
        "dorothea_nips03",
        "gcm_ramaswamy",
        "gli_85",
        "glioma_50_4class",
        "hf_breast_ge_mubashir1837",
        "leukemia_1_72_3class",
        "leukemia_golub",
        "lymphoma_3",
        "lymphoma_9",
        "madelon_nips03",
        "nci60_strict_holdout",
        "nci9_60_9class",
        "ovarian_petricoin",
        "prostate_singh",
        "srbct_khan",
        "tox_171",
        "tumor11_su",
        "xena_tcga_coad_cms",
        "xena_tcga_gbm",
        "xena_tcga_lgg",
        "xena_tcga_skcm",
        # Gap-filling extension (5)
        "nci_61_8class",
        "tumor9_openml",
        "cumida_gastric_gse54129",
        "cumida_renal_gse53757",
        "breast_vantveer",
    ]

    fs_method_sets = _load_fs_method_sets()
    benchmark_datasets, _dataset_sets = _load_benchmark_registry()
    hf_meta = _load_hf_manifest_metadata()
    hf_ids = set(hf_meta.keys())
    if not hf_ids:
        raise RuntimeError("HF bundle manifest metadata is empty.")

    validation_ids = [ds_id for ds_id in VAL13_DATASETS if ds_id in hf_ids]
    if not validation_ids:
        raise RuntimeError("No Val-13 clf-oracle datasets available in HF bundle manifests.")
    missing_hf = [ds_id for ds_id in VAL13_DATASETS if ds_id not in hf_ids]
    if missing_hf:
        print(
            f"[validation13_clf_oracle] Skipping {len(missing_hf)} dataset(s) not in HF: "
            f"{', '.join(missing_hf[:10])}{'...' if len(missing_hf) > 10 else ''}",
            file=sys.stderr,
        )

    for required in ("mnpo_broad_all",):
        if required not in fs_method_sets:
            raise RuntimeError(f"Missing required method set {required!r} in FS_METHOD_SETS.")

    # --- Runtime estimation (use Val-12 d_default as base) ---
    if val12_root is None:
        val12_root = (
            REPO_ROOT
            / "run_artifacts"
            / "validation-12"
            / "val12_4pods_live_20260228_013532"
        )
    val12_runtime = _load_runtime_hints_from_summaries(val12_root, phase_tag="val12")
    val12_d_hist = dict(val12_runtime.get("d_default") or {})
    val12_a_hist = dict(val12_runtime.get("a_control") or {})

    legacy8_est: Dict[str, float] = {}
    legacy16_est: Dict[str, float] = {}
    mnpo8_est: Dict[str, float] = {}
    mnpo16_est: Dict[str, float] = {}

    for ds_id in validation_ids:
        spec = benchmark_datasets.get(ds_id)
        d_seed = float(val12_d_hist.get(ds_id, 0.0))
        if d_seed <= 0.0:
            base_t = float(val12_a_hist.get(ds_id, 0.0))
            if base_t <= 0.0:
                base_t = float(max(60.0, _dataset_weight(spec) * 220.0))
            d_seed = float(base_t * 1.35)
        # Legacy+8clf ≈ d_default cost (same FS + legacy clf with 8 candidates)
        legacy8_est[ds_id] = float(d_seed)
        # Legacy+16clf: more CV candidates → ~20% slower classifier stage
        legacy16_est[ds_id] = float(d_seed * 1.12)
        # MNPO+8clf: MNPO oracle overhead with 8 candidates ≈ +15%
        mnpo8_est[ds_id] = float(d_seed * 1.15)
        # MNPO+16clf: MNPO oracle + 16 candidates + FLAML HPO ≈ +30%
        mnpo16_est[ds_id] = float(d_seed * 1.30)

    ds_items: List[Tuple[str, float]] = []
    for ds_id in validation_ids:
        combined = float(
            (
                legacy8_est[ds_id]
                + legacy16_est[ds_id]
                + mnpo8_est[ds_id]
                + mnpo16_est[ds_id]
            )
            * len(VALIDATION_SEEDS)
        )
        ds_items.append((str(ds_id), combined))
    ds_parts = _balanced_partition(ds_items, int(max(1, dataset_shards)))

    common_extra_args: Tuple[str, ...] = ("--emit-summary", "--compute-budget", "standard")

    # Shared stage flags (identical to Val-13 main)
    shared_stage_flags: Tuple[str, ...] = (
        "--df-family-set", "flex",
        "--df-compute-ad",
        "--df-compute-qq-pp",
        "--df-compute-dip",
        "--df-interval-likelihood",
        "--df-compute-crps",
        "--df-crps-uq-decomposition",
        "--df-lmoment-prescreen",
        "--df-lmoment-prescreen-max-candidates", "12",
        "--folding-method", "pls_da",
        "--enable-prefilter-rnaseq-nb-lrt",
        "--prefilter-rnaseq-nb-lrt-alpha", "0.10",
        "--enable-classifier-conformal",
        "--classifier-conformal-alpha", "0.10",
        "--classifier-conformal-calibration-fraction", "0.25",
        "--classifier-conformal-min-calibration", "20",
        "--enable-stage2-ratio-augmentation",
        "--stage2-ratio-max-features", "16",
        "--stage2-ratio-selection-method", "correlation",
        "--enable-model-cv-runtime-containment",
        "--stage2-max-train-test-gap", "0.15",
        "--stage2-tree-complexity-penalty-enabled",
        "--stage2-tree-complexity-penalty-strength", "0.1",
    )

    # Shared MNPO FS flags (d_default style, identical across all 4 profiles)
    shared_fs_flags: Tuple[str, ...] = (
        *shared_stage_flags,
        "--dist-criterion", "simple",
        "--enable-fs-adaptive-portfolio-sizing",
        "--fs-adaptive-size-min", "4",
        "--fs-adaptive-size-max", "8",
        "--fs-portfolio-size-guard", "warn",
        "--enable-diversity-oracle",
        "--enable-fs-mrmr-mi-redundancy",
        "--prefilter-union-enabled",
        "--prefilter-wsnr-enabled",
        "--prefilter-strategies", "mi_ftest_blend,rf_importance,wsnr",
        "--screening-enabled",
        "--screening-method", "evalue",
        "--eval-models-enabled",
        "--mnpo-performance-oracle-mode", "multi_model_oracles",
        "--eval-models", "lr_l2,linear_svc,rf_small",
        "--eval-aggregate", "mean",
        "--fs-copula-derandomize-runs", "3",
        "--fs-diversity-oracle-mode", "legacy_jaccard",
        "--fs-oracle-weighting-mode", "banzhaf",
        "--fs-shapley-n-coalitions-max", "2048",
    )

    # Classifier candidate pools
    pool8_clf_flags: Tuple[str, ...] = (
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
        "--model-cv-runtime-max-candidates", "8",
    )

    pool16_clf_flags: Tuple[str, ...] = (
        "--model-candidates",
        "lr", "svm_rbf", "svm_linear", "dlda", "shrinkage_lda",
        "nsc", "pls_da_classifier", "nb", "vote_ensemble",
        "elastic_net_lr", "rf", "knn", "xgb", "lgbm", "extra_tree", "catboost",
        "--include-nsc-model",
        "--include-pls-da-model",
        "--include-lgbm-model",
        "--include-extra-tree-model",
        "--include-catboost-model",
        "--model-cv-runtime-max-candidates", "16",
    )

    # Selection algorithm flags
    legacy_clf_flags: Tuple[str, ...] = (
        "--classifier-selection-mode", "legacy",
        "--classification-backend", "sklearn",
    )

    mnpo_clf_flags: Tuple[str, ...] = (
        "--classifier-selection-mode", "mnpo_hybrid",
        "--classifier-oracle-k", "2",
        "--classifier-oracle-weighting-mode", "tritrust",
        "--classifier-oracle-bbc-bootstrap-rounds", "120",
        "--classifier-oracle-bbc-ci-level", "0.90",
        "--enable-classifier-oracle-ensemble",
        "--classification-backend", "flaml",
        "--flaml-time-budget", "120",
    )

    # Build the 4 profile flag tuples
    clf_legacy_pool8_flags: Tuple[str, ...] = (
        *shared_fs_flags, *legacy_clf_flags, *pool8_clf_flags,
    )
    clf_legacy_pool16_flags: Tuple[str, ...] = (
        *shared_fs_flags, *legacy_clf_flags, *pool16_clf_flags,
    )
    clf_mnpo_pool8_flags: Tuple[str, ...] = (
        *shared_fs_flags, *mnpo_clf_flags, *pool8_clf_flags,
    )
    clf_mnpo_pool16_flags: Tuple[str, ...] = (
        *shared_fs_flags, *mnpo_clf_flags, *pool16_clf_flags,
    )

    common_job_params = dict(
        seeds=list(VALIDATION_SEEDS),
        ablation_profile="none",
        allow_synthetic_fallback=False,
        dataset_integrity_policy="skip",
        quiet_worker_logs=True,
        progress_heartbeat_sec=30,
        progress_watchdog_sec=0,
        progress_stall_watchdog_sec=1800,
        task_timeout_sec=21600,
        fs_method_timeout_sec=3600,
    )

    profiles: List[BenchmarkProfile] = [
        BenchmarkProfile(
            profile_id="clf_legacy_pool8",
            fs_method_set="mnpo_broad_all",
            extra_args=clf_legacy_pool8_flags,
            notes=(
                "Val-13 clf-oracle 2×2: MNPO FS + legacy CV classifier + 8-candidate pool. "
                "Baseline cell (legacy selection, small pool)."
            ),
        ),
        BenchmarkProfile(
            profile_id="clf_legacy_pool16",
            fs_method_set="mnpo_broad_all",
            extra_args=clf_legacy_pool16_flags,
            notes=(
                "Val-13 clf-oracle 2×2: MNPO FS + legacy CV classifier + 16-candidate pool. "
                "Tests pool-size effect under legacy selection."
            ),
        ),
        BenchmarkProfile(
            profile_id="clf_mnpo_pool8",
            fs_method_set="mnpo_broad_all",
            extra_args=clf_mnpo_pool8_flags,
            notes=(
                "Val-13 clf-oracle 2×2: MNPO FS + MNPO hybrid classifier + 8-candidate pool. "
                "Tests selection-algorithm effect with matched (small) pool."
            ),
        ),
        BenchmarkProfile(
            profile_id="clf_mnpo_pool16",
            fs_method_set="mnpo_broad_all",
            extra_args=clf_mnpo_pool16_flags,
            notes=(
                "Val-13 clf-oracle 2×2: MNPO FS + MNPO hybrid classifier + 16-candidate pool. "
                "Full MNPO classifier pathway (oracle + ensemble + FLAML HPO, large pool)."
            ),
        ),
    ]

    deprecated = set(_load_deprecated_method_sets(sorted(fs_method_sets.keys())))
    invalid: List[str] = []
    for prof in profiles:
        if prof.fs_method_set not in fs_method_sets:
            invalid.append(f"{prof.profile_id}: unknown fs_method_set={prof.fs_method_set!r}")
        if prof.fs_method_set in deprecated:
            invalid.append(f"{prof.profile_id}: deprecated fs_method_set={prof.fs_method_set!r}")
    if invalid:
        raise RuntimeError("Invalid clf-oracle profile(s):\n- " + "\n- ".join(invalid))

    runtime_by_profile = {
        "clf_legacy_pool8": legacy8_est,
        "clf_legacy_pool16": legacy16_est,
        "clf_mnpo_pool8": mnpo8_est,
        "clf_mnpo_pool16": mnpo16_est,
    }

    jobs: List[Job] = []
    for prof in profiles:
        profile_runtime = dict(runtime_by_profile.get(prof.profile_id) or {})
        for part_idx, ds_list in enumerate(ds_parts, start=1):
            part_seed_sec = 0.0
            for ds_id in ds_list:
                part_seed_sec += float(profile_runtime.get(ds_id, 180.0))
            part_weight = float(part_seed_sec * len(VALIDATION_SEEDS))
            jobs.append(
                _job(
                    f"val13_clf/{prof.profile_id}/ds{part_idx:02d}",
                    "run_df_fs_sota_benchmark",
                    weight=part_weight,
                    fs_method_set=prof.fs_method_set,
                    datasets=list(ds_list),
                    extra_args=list(common_extra_args + tuple(prof.extra_args)),
                    **common_job_params,
                )
            )
    return jobs


def _balanced_shard_assign_validation13_clf_oracle_bundles(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-13 clf-oracle by dataset partitions (group all 4 profiles per partition)."""
    _CLF_ORACLE_PROFILES = {
        "clf_legacy_pool8",
        "clf_legacy_pool16",
        "clf_mnpo_pool8",
        "clf_mnpo_pool16",
    }
    profile_re = re.compile(
        r"^val13_clf/(" + "|".join(re.escape(p) for p in _CLF_ORACLE_PROFILES) + r")/(ds\d+)$"
    )
    grouped: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = profile_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        grouped.setdefault(part_id, {})[profile_id] = job

    bundle_items: List[Tuple[float, List[Job]]] = []
    for part_id, bundle in grouped.items():
        bundle_weight = sum(float(j.weight) for j in bundle.values())
        bundle_items.append((bundle_weight, list(bundle.values())))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for bundle_weight, bundle_jobs in sorted(bundle_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        for j in sorted(bundle_jobs, key=lambda j: j.job_id):
            shards[target].append(j.job_id)
        totals[target] += float(bundle_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


def _load_runtime_hints_from_summaries(root: Path, *, phase_tag: str) -> Dict[str, Dict[str, float]]:
    """Load mean per-seed runtime from run_summary files for a validation phase.

    Supported layouts:
    - ``<root>/shard*/<phase_tag>/<profile>/<ds_part>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json``
    - ``<root>/<phase_tag>/<profile>/<ds_part>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json``
    - ``<root>/<host>/<phase_tag>/<profile>/<ds_part>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json``
    """
    import glob as _glob

    patterns = (
        str(root / "shard*" / phase_tag / "*" / "*" / "*_df_fs_sota_benchmark" / "run_summary_v1.json"),
        str(root / phase_tag / "*" / "*" / "*_df_fs_sota_benchmark" / "run_summary_v1.json"),
        str(root / "*" / phase_tag / "*" / "*" / "*_df_fs_sota_benchmark" / "run_summary_v1.json"),
    )
    buckets: Dict[Tuple[str, str], List[float]] = {}

    seen_paths: Set[str] = set()
    for pattern in patterns:
        for fpath in _glob.glob(pattern):
            if fpath in seen_paths:
                continue
            try:
                data = json.loads(Path(fpath).read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            seen_paths.add(str(fpath))

            marker = f"/{phase_tag}/"
            parts = str(fpath).split(marker)
            if len(parts) < 2:
                continue
            profile_id = parts[1].split("/")[0]
            if not profile_id:
                continue

            for r in data.get("results", []):
                ds_id = str(r.get("dataset_id", "")).strip()
                if not ds_id:
                    continue
                rt = r.get("runtime") or {}
                try:
                    total_t = (
                        float(rt.get("fs_time_sec", 0) or 0)
                        + float(rt.get("dist_time_sec", 0) or 0)
                        + float(rt.get("transform_time_sec", 0) or 0)
                    )
                except Exception:
                    continue
                if total_t > 0:
                    buckets.setdefault((profile_id, ds_id), []).append(float(total_t))

    out: Dict[str, Dict[str, float]] = {}
    for (profile_id, ds_id), vals in buckets.items():
        if not vals:
            continue
        out.setdefault(profile_id, {})[ds_id] = float(sum(vals) / len(vals))
    return out


def _load_val6_runtime_hints_from_summaries(val6_root: Path) -> Dict[str, Dict[str, float]]:
    return _load_runtime_hints_from_summaries(val6_root, phase_tag="val6")


def _load_val7_runtime_hints_from_summaries(val7_root: Path) -> Dict[str, Dict[str, float]]:
    return _load_runtime_hints_from_summaries(val7_root, phase_tag="val7")


def _load_val8_runtime_hints_from_summaries(val8_root: Path) -> Dict[str, Dict[str, float]]:
    return _load_runtime_hints_from_summaries(val8_root, phase_tag="val8")


def _load_val9_runtime_hints_from_summaries(val9_root: Path) -> Dict[str, Dict[str, float]]:
    return _load_runtime_hints_from_summaries(val9_root, phase_tag="val9")


def _balanced_shard_assign_validation7_pairs(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-7 by paired dataset partitions (baseline + candidate)."""
    pair_re = re.compile(r"^val7/(baseline|candidate)/(ds\d+)$")
    paired: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = pair_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        paired.setdefault(part_id, {})[profile_id] = job

    pair_items: List[Tuple[float, Job, Job]] = []
    for part_id, bundle in paired.items():
        baseline_job = bundle.get("baseline")
        candidate_job = bundle.get("candidate")
        if baseline_job is None or candidate_job is None:
            for partial in bundle.values():
                unpaired.append(partial)
            continue
        pair_weight = float(baseline_job.weight) + float(candidate_job.weight)
        pair_items.append((pair_weight, candidate_job, baseline_job))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for pair_weight, candidate_job, baseline_job in sorted(pair_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        shards[target].append(candidate_job.job_id)
        shards[target].append(baseline_job.job_id)
        totals[target] += float(pair_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


def _balanced_shard_assign_validation8_pairs(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-8 by paired dataset partitions (baseline + candidate)."""
    pair_re = re.compile(r"^val8/(baseline|candidate)/(ds\d+)$")
    paired: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = pair_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        paired.setdefault(part_id, {})[profile_id] = job

    pair_items: List[Tuple[float, Job, Job]] = []
    for part_id, bundle in paired.items():
        baseline_job = bundle.get("baseline")
        candidate_job = bundle.get("candidate")
        if baseline_job is None or candidate_job is None:
            for partial in bundle.values():
                unpaired.append(partial)
            continue
        pair_weight = float(baseline_job.weight) + float(candidate_job.weight)
        pair_items.append((pair_weight, candidate_job, baseline_job))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for pair_weight, candidate_job, baseline_job in sorted(pair_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        shards[target].append(candidate_job.job_id)
        shards[target].append(baseline_job.job_id)
        totals[target] += float(pair_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


def _balanced_shard_assign_validation9_pairs(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-9 by paired dataset partitions (legacy_full + mnpo_hybrid)."""
    pair_re = re.compile(r"^val9/(legacy_full|mnpo_hybrid)/(ds\d+)$")
    paired: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = pair_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        paired.setdefault(part_id, {})[profile_id] = job

    pair_items: List[Tuple[float, Job, Job]] = []
    for _, bundle in paired.items():
        legacy_job = bundle.get("legacy_full")
        hybrid_job = bundle.get("mnpo_hybrid")
        if legacy_job is None or hybrid_job is None:
            for partial in bundle.values():
                unpaired.append(partial)
            continue
        pair_weight = float(legacy_job.weight) + float(hybrid_job.weight)
        pair_items.append((pair_weight, hybrid_job, legacy_job))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for pair_weight, hybrid_job, legacy_job in sorted(pair_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        shards[target].append(hybrid_job.job_id)
        shards[target].append(legacy_job.job_id)
        totals[target] += float(pair_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


def _balanced_shard_assign_validation10_pairs(
    jobs: Sequence[Job], num_shards: int
) -> Dict[int, List[str]]:
    """Shard Val-10 by paired dataset partitions (simple_all_stages + mnpo_all_stages)."""
    pair_re = re.compile(r"^val10/(simple_all_stages|mnpo_all_stages)/(ds\d+)$")
    paired: Dict[str, Dict[str, Job]] = {}
    unpaired: List[Job] = []

    for job in jobs:
        m = pair_re.match(str(job.job_id))
        if m is None:
            unpaired.append(job)
            continue
        profile_id, part_id = str(m.group(1)), str(m.group(2))
        paired.setdefault(part_id, {})[profile_id] = job

    pair_items: List[Tuple[float, Job, Job]] = []
    for _, bundle in paired.items():
        simple_job = bundle.get("simple_all_stages")
        mnpo_job = bundle.get("mnpo_all_stages")
        if simple_job is None or mnpo_job is None:
            for partial in bundle.values():
                unpaired.append(partial)
            continue
        pair_weight = float(simple_job.weight) + float(mnpo_job.weight)
        pair_items.append((pair_weight, mnpo_job, simple_job))

    shards: Dict[int, List[str]] = {i: [] for i in range(1, num_shards + 1)}
    totals: Dict[int, float] = {i: 0.0 for i in range(1, num_shards + 1)}

    for pair_weight, mnpo_job, simple_job in sorted(pair_items, key=lambda x: float(x[0]), reverse=True):
        target = min(totals.items(), key=lambda kv: kv[1])[0]
        shards[target].append(mnpo_job.job_id)
        shards[target].append(simple_job.job_id)
        totals[target] += float(pair_weight)

    if unpaired:
        for job in sorted(unpaired, key=lambda j: float(j.weight), reverse=True):
            target = min(totals.items(), key=lambda kv: kv[1])[0]
            shards[target].append(job.job_id)
            totals[target] += float(job.weight)

    return shards


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_work_split_md(path: Path, *, shards: Dict[int, List[str]], jobs_by_id: Dict[str, Job]) -> None:
    lines: List[str] = []
    lines.append("# Work Split\n")
    lines.append("")
    lines.append("Each shard is intended to run on exactly one pod.")
    lines.append("")
    recommended_host_assignment: Dict[str, Any] = {}
    host_order: Tuple[str, ...] = tuple()
    all_job_ids = [jid for shard_job_ids in shards.values() for jid in shard_job_ids]
    if all_job_ids and all(str(jid).startswith("val20_tabarena_") for jid in all_job_ids):
        recommended_host_assignment = _validation20_tabarena_recommended_host_assignment(shards, jobs_by_id)
        host_order = tuple(VAL20_TABARENA_HOSTS)
    elif all_job_ids and all(str(jid).startswith("val20_") for jid in all_job_ids):
        recommended_host_assignment = _validation20_recommended_host_assignment(shards, jobs_by_id)
        host_order = tuple((*VAL20_CPU_HOSTS, *VAL20_GPU_HOSTS))
    elif all_job_ids and all(str(jid).startswith("val19_") for jid in all_job_ids):
        recommended_host_assignment = _validation19_recommended_host_assignment(shards, jobs_by_id)
        host_order = tuple(VAL18_CPU_HOSTS)
    elif all_job_ids and all(
        str(jid).startswith("val18_") or str(jid).startswith("val19_")
        for jid in all_job_ids
    ):
        recommended_host_assignment = _validation18_recommended_host_assignment(shards, jobs_by_id)
        host_order = tuple((*VAL18_CPU_HOSTS, *VAL18_GPU_HOSTS))
    if recommended_host_assignment:
        host_summary = dict(recommended_host_assignment.get("host_summary") or {})
        if host_summary:
            lines.append("## Recommended Host Assignment")
            lines.append("")
            for host in host_order:
                summary = dict(host_summary.get(host) or {})
                assigned_shards = [int(s) for s in list(summary.get("shards") or [])]
                worker_target = dict(summary.get("worker_target") or {})
                shards_txt = ", ".join(f"`{sid}`" for sid in assigned_shards) if assigned_shards else "none"
                bits: List[str] = []
                if worker_target:
                    bits.append(f"`PODS_PER_HOST={int(worker_target.get('pods_per_host', 0) or 0)}`")
                    bits.append(f"`MAX_WORKERS={int(worker_target.get('max_workers_per_pod', 0) or 0)}`")
                    bits.append(f"`target_total_workers={int(worker_target.get('target_total_workers', 0) or 0)}`")
                norm = float(summary.get("normalized_weight_per_core", 0.0) or 0.0)
                line = f"- `{host}`: shards {shards_txt}"
                if bits:
                    line += "; " + ", ".join(bits)
                if assigned_shards:
                    line += f"; normalized shard weight/core `{norm:.1f}`"
                line += "."
                lines.append(line)
            lines.append("")
    for shard_id in sorted(shards.keys()):
        shard_job_ids = list(shards[shard_id])
        shard_has_hsic = any("hsic_lasso" in jid for jid in shard_job_ids)
        shard_has_tabpfn = False
        shard_has_eval_proxy = False
        shard_has_screening = False
        shard_has_df_mnpo_oracle = False
        shard_has_pls_da = False
        shard_worker_targets: List[Tuple[str, int]] = []
        shard_execution_lanes: Set[str] = set()
        shard_dataset_panels: Set[str] = set()
        shard_preferred_hosts: Set[str] = set()

        for jid in shard_job_ids:
            job = jobs_by_id.get(jid)
            if job is None:
                continue

            extra = list(job.params.get("extra_args") or [])
            if "--include-tabpfn-model" in extra or "tabpfn" in str(jid):
                shard_has_tabpfn = True
            lane = _job_execution_lane(job)
            if lane:
                shard_execution_lanes.add(lane)
            panel = str(dict(job.params or {}).get("dataset_panel", "") or "").strip().lower()
            if panel:
                shard_dataset_panels.add(panel)
            for host_txt in _job_preferred_hosts(job):
                shard_preferred_hosts.add(str(host_txt))
            if "--eval-models-enabled" in extra:
                shard_has_eval_proxy = True
            if "--screening-enabled" in extra:
                shard_has_screening = True
            if "--dist-criterion" in extra:
                try:
                    idx = extra.index("--dist-criterion")
                    if idx + 1 < len(extra) and str(extra[idx + 1]).strip().lower() == "mnpo_oracle":
                        shard_has_df_mnpo_oracle = True
                except ValueError:
                    pass
            if "--folding-method" in extra:
                try:
                    idx = extra.index("--folding-method")
                    if idx + 1 < len(extra) and str(extra[idx + 1]).strip().lower() == "pls_da":
                        shard_has_pls_da = True
                except ValueError:
                    pass
            if job.kind == "validation_suite":
                if str(job.params.get("df_criterion", "") or "").strip().lower() == "mnpo_oracle":
                    shard_has_df_mnpo_oracle = True
            if job.kind == "run_df_fs_sota_benchmark":
                raw_workers = job.params.get("max_workers")
                if raw_workers is not None:
                    try:
                        workers = int(raw_workers)
                    except Exception:
                        workers = 0
                    if workers > 0:
                        shard_worker_targets.append((jid, workers))
        lines.append(f"## Shard {shard_id}")
        has_explicit_workers = bool(shard_worker_targets)
        if shard_has_eval_proxy:
            if has_explicit_workers:
                lines.append("- Resources: CPU-heavy. RAM: 64 GB recommended.")
            else:
                lines.append("- Resources: CPU-heavy. RAM: 64 GB recommended. Suggested `MAX_WORKERS<=4` (set `MAX_WORKERS=2` if <=32 GB).")
        elif shard_has_hsic:
            if has_explicit_workers:
                lines.append("- Resources: CPU-heavy. RAM: 64 GB recommended (HSIC Lasso present).")
            else:
                lines.append("- Resources: CPU-heavy. RAM: 64 GB recommended (HSIC Lasso present). Suggested `MAX_WORKERS<=2` (set `MAX_WORKERS=1` if <=32 GB).")
        elif shard_has_df_mnpo_oracle or shard_has_screening or shard_has_pls_da:
            if has_explicit_workers:
                lines.append("- Resources: CPU-medium. RAM: 32 GB recommended.")
            else:
                lines.append("- Resources: CPU-medium. RAM: 32 GB recommended. Suggested `MAX_WORKERS<=4` (16 GB OK for `MAX_WORKERS<=2`).")
        else:
            if shard_has_tabpfn:
                if has_explicit_workers:
                    lines.append("- Resources: GPU recommended (TabPFN present); CPU-only will work but may be slow. RAM: 32 GB recommended.")
                else:
                    lines.append("- Resources: GPU recommended (TabPFN present); CPU-only will work but may be slow. RAM: 32 GB recommended. Suggested `MAX_WORKERS<=4`.")
            else:
                if has_explicit_workers:
                    lines.append("- Resources: CPU-only (no GPU). RAM: 32 GB recommended.")
                else:
                    lines.append("- Resources: CPU-only (no GPU). RAM: 32 GB recommended. Suggested `MAX_WORKERS<=4` (16 GB OK for `MAX_WORKERS<=2`).")
        if shard_execution_lanes:
            lines.append("- Execution lane: " + ", ".join(f"`{lane}`" for lane in sorted(shard_execution_lanes)) + ".")
        if shard_dataset_panels:
            lines.append("- Dataset panel: " + ", ".join(f"`{panel}`" for panel in sorted(shard_dataset_panels)) + ".")
        if shard_preferred_hosts:
            lines.append("- Preferred hosts: " + ", ".join(f"`{host}`" for host in sorted(shard_preferred_hosts)) + ".")
        if recommended_host_assignment:
            shard_to_host = dict(recommended_host_assignment.get("shard_to_host") or {})
            recommended_host = str(shard_to_host.get(str(shard_id), "") or "").strip()
            if recommended_host:
                lines.append(f"- Recommended host: `{recommended_host}`.")
        if shard_worker_targets:
            targets_txt = ", ".join(f"`{jid}`→`{workers}`" for jid, workers in shard_worker_targets)
            lines.append(
                "- Plan worker targets: "
                f"{targets_txt}. `run_shard.py` applies these per-job values; "
                "if `MAX_WORKERS` is set, it acts as a cap."
            )
        for job_id in shards[shard_id]:
            job = jobs_by_id.get(job_id)
            kind = job.kind if job is not None else "unknown"
            lines.append(f"- `{job_id}` ({kind})")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a sharded validation plan for RunPod.")
    parser.add_argument("--num-pods", type=int, default=DEFAULT_MAX_PODS, help="Shard count to generate (cap: 20).")
    parser.add_argument("--max-pods", type=int, default=DEFAULT_MAX_PODS, help="Hard cap for ideal pod count reporting.")
    parser.add_argument(
        "--plan-kind",
        type=str,
        default="validation1",
        choices=[
            "validation1",
            "validation4",
            "validation5",
            "validation6",
            "validation7",
            "validation8",
            "validation9",
            "validation10",
            "validation11",
            "validation12",
            "validation13",
            "validation13_clf_oracle",
            "validation14",
            "validation14_activation_smoke",
            "validation15",
            "validation16",
            "validation17",
            "validation18_anchors",
            "validation18_singletons",
            "validation18_mnpo",
            "validation18_stage",
            "validation18_classifiers",
            "validation18_cls_oracle_wt",
            "validation19_classifiers",
            "validation20_wave1",
            "validation20_ensemble",
            "validation20_tabarena_w1",
            "validation20_tabarena_w2",
            "validation20_tabarena_w3",
            "legacy",
        ],
        help=(
            "Plan kind: validation17 (Val-16 matrix rerun on the packaged modular architecture with "
            "df_stage_position=after_fs and dataset_integrity_policy=error); "
            "Plan kind: validation16 (post-Val-15 focused matrix with 11 profiles: "
            "a_control, d_default, v16_ref, v16_clp, v16_payoff_shrink, v16_clp_shrink, "
            "v16_conformal_eff, v16_js_shrinkage, v16_meta_dt, v16_multiomics, "
            "v16_full_stack over the 64-dataset expanded catalog); "
            "Plan kind: validation15 (post-Val-14 focused matrix with 9 profiles: "
            "a_control, d_default, v15_ref_ipss, v15_no_ipss, v15_no_regime_fallback, "
            "v15_mapie_aps/raps/cross, v15_multiomics_adapter; low-p/n bypass disabled, "
            "very-hard fallback retained); "
            "Plan kind: validation14 (Val-14 redesigned 20-profile feature-effect matrix over the same "
            "35-dataset catalog: anchors + one-feature perturbations + activation-ready contrasts); "
            "Plan kind: validation14_activation_smoke (Val-14 pre-launch activation smoke: same profiles on "
            "6 focused datasets with 1 seed for hard activation gating); "
            "Plan kind: validation13 (Val-13 profile matrix + 5-fraction FS sweep: "
            "a_control, d_default, d_v13, d_v13_pareto, d_v13_multiclass, g_alt_screening, "
            "d_v13_fs025/fs030/fs040/fs050/fs055 over 35 datasets); "
            "Plan kind: validation12 (regime-conditional gating + oracle/fs-fraction profile sweep: "
            "a_control, d_default, d_gated, d_gated_cvar, d_gated_oracle_slim, "
            "d_gated_oracle_slim_fs030, d_gated_oracle_slim_fs055 over Val-11 core + failure-regime expansion); "
            "Plan kind: validation11 (3-profile MNPO improvement ablation: "
            "simple_control vs mnpo_improved vs mnpo_val10_ref, tests Banzhaf oracle "
            "weighting + oracle pruning + selector penalty + train/test gap gating); "
            "Plan kind: validation10 (extended-catalog paired comparison of simple-vs-MNPO "
            "usage across DF/FS/classifier stages, includes rv_* datasets); "
            "Plan kind: validation9 (comprehensive full-stack run, classifier selection "
            "legacy_full vs mnpo_hybrid, 67-dataset extended catalog, Val-8 runtime hints); "
            "Plan kind: validation8 (Val-7 candidate + RNA-seq NB-LRT prefilter + "
            "Stage-2 classifier backend support, baseline vs candidate, 67-dataset "
            "extended catalog, Val-7 runtime hints); "
            "Plan kind: validation7 (classifier upgrade + shrinkage, baseline vs candidate, "
            "67-dataset extended catalog, Val-6 runtime hints); "
            "Plan kind: validation18_anchors (Val-18 anchor/control baseline sweep: "
            "8 profiles x 64 datasets x 9 seeds, families A01-A03 + skip-FS controls); "
            "Plan kind: validation18_singletons (Val-18 singleton method sweep: "
            "78 profiles (39 methods x raw/scaffold) x 64 datasets x 5 seeds); "
            "Plan kind: validation18_mnpo (Val-18 MNPO/oracle ablation: "
            "25 profiles x 64 datasets x 5 seeds, including oracle-pruning and 5x5 fold-resolution sweeps); "
            "Plan kind: validation18_stage (Val-18 DF/prefilter/scaffold matrix: "
            "40 profiles (D+P+S01-S03) x 24 diagnostic datasets x 5 seeds); "
            "Plan kind: validation18_classifiers (Val-18 classifier universe sweep: "
            "auto-generated DIAG24 C_ONLY profiles plus 3 FULL64 TabPFN reruns, sharded into CPU and TabPFN lanes); "
            "Plan kind: validation19_classifiers (Val-19 added-classifier extension: "
            "7 DIAG24 singleton diagnostics, 4 matched FULL64 old-vs-new CPU-pool profiles, "
            "and 2 FULL64 val18-compat classifier-oracle controls); "
            "Plan kind: validation20_wave1 (Val-20 Wave 1 decision campaign: "
            "bridge anchors, FLAML budget/custom diagnostics, tune-first ablations, "
            "TabPFN disposition, prefilter/ordering tests, LR-diversity tests, and "
            "the Family-E ensemble/oracle enhancement diagnostics; evidence-driven "
            "main-validation follow-ons F05/F06 and C01-C04 stay reserved until Wave 1 picks "
            "their concrete settings); "
            "Plan kind: validation20_tabarena_w1/w2/w3 (Val-20 TabArena companion lane: "
            "Wave 1 probe refresh + competitive challenger, Wave 2 full-catalog official reruns, "
            "Wave 3 final chase among baseline and top challengers); "
            "validation6 (2-profile signal check on "
            "the 67-dataset Val-5 catalog, baseline vs full); "
            "validation5 (baseline vs complete B/C/D candidate, "
            "6 pods, runtime-informed sharding); "
            "validation4 (3-profile decomposition + PLS-DA guardrail, 6 pods); "
            "validation1 (feature ablations, 8+ pods); "
            "legacy (full method-set sweeps)."
        ),
    )
    parser.add_argument(
        "--dataset-shards",
        type=int,
        default=8,
        help=(
            "Number of dataset partitions per profile "
            "(validation1/validation4/validation5/validation6/validation7/validation8/validation9/"
            "validation10/validation11/validation12/validation13/validation14/validation14_activation_smoke/"
            "validation15/validation16/validation17/validation18_*/validation19_classifiers/"
            "validation20_wave1/validation20_ensemble/validation20_tabarena_w1/"
            "validation20_tabarena_w2/validation20_tabarena_w3)."
        ),
    )
    parser.add_argument(
        "--val4-root",
        type=str,
        default="",
        help=(
            "(validation5) Path to Val-4 merged output root containing "
            "_aggregate/benchmark_df_fs_runs__all_jobs.csv for runtime hints."
        ),
    )
    parser.add_argument(
        "--val5-root",
        type=str,
        default="",
        help=(
            "(validation6) Path to Val-5 merged output root containing "
            "_aggregate/benchmark_df_fs_runs__all_jobs.csv for runtime hints."
        ),
    )
    parser.add_argument(
        "--val6-root",
        type=str,
        default="",
        help=(
            "(validation7) Path to Val-6 finished pull directory containing "
            "shard*/val6/<profile>/<ds>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json "
            "for runtime hints."
        ),
    )
    parser.add_argument(
        "--val7-root",
        type=str,
        default="",
        help=(
            "(validation8) Path to Val-7 merged output directory containing "
            "shard*/val7/<profile>/<ds>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json "
            "for runtime hints."
        ),
    )
    parser.add_argument(
        "--val8-root",
        type=str,
        default="",
        help=(
            "(validation9) Path to Val-8 merged output directory containing "
            "shard*/val8/<profile>/<ds>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json "
            "for runtime hints."
        ),
    )
    parser.add_argument(
        "--val9-root",
        type=str,
        default="",
        help=(
            "(validation10) Path to Val-9 output directory containing "
            "shard*/val9/<profile>/<ds>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json "
            "for runtime hints."
        ),
    )
    parser.add_argument(
        "--val10-root",
        type=str,
        default="",
        help=(
            "(validation11) Path to Val-10 output directory containing "
            "val10/<profile>/<ds>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json "
            "for runtime hints."
        ),
    )
    parser.add_argument(
        "--val11-root",
        type=str,
        default="",
        help=(
            "(validation12) Path to Val-11 output directory containing "
            "val11/<profile>/<ds>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json "
            "for runtime hints."
        ),
    )
    parser.add_argument(
        "--val12-root",
        type=str,
        default="",
        help=(
            "(validation13) Path to Val-12 output directory containing "
            "val12/<profile>/<ds>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json "
            "for runtime hints."
        ),
    )
    parser.add_argument(
        "--val13-root",
        type=str,
        default="",
        help=(
            "(validation14/validation14_activation_smoke) Path to Val-13 output directory containing "
            "val13/<profile>/<ds>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json "
            "for runtime hints."
        ),
    )
    parser.add_argument(
        "--val14-root",
        type=str,
        default="",
        help=(
            "(validation15) Path to Val-14 output directory containing "
            "val14/<profile>/<ds>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json "
            "for runtime hints."
        ),
    )
    parser.add_argument(
        "--val15-root",
        type=str,
        default="",
        help=(
            "(validation16/validation17) Path to Val-15 output directory containing "
            "val15/<profile>/<ds>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json "
            "for runtime hints."
        ),
    )
    parser.add_argument(
        "--val17-root",
        type=str,
        default="",
        help=(
            "(validation18_*/validation19_classifiers/validation20_*) Path to Val-17 output directory containing "
            "val17/<profile>/<ds>/<timestamp>_df_fs_sota_benchmark/run_summary_v1.json "
            "for runtime hints."
        ),
    )
    parser.add_argument(
        "--val9-runtime-profile",
        type=str,
        default="full",
        choices=["full", "tuned"],
        help=(
            "(validation9) Runtime profile for Val-9 plan generation. "
            "'full' keeps original budgets; 'tuned' reduces expensive budget knobs "
            "(Shapley coalitions, UBayFS bootstraps, FLAML/model-CV budgets, timeouts) "
            "while preserving method/oracle coverage."
        ),
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help=(
            "Optional suffix appended to output filenames. "
            "Example: --output-tag val5 writes plan_<N>_val5.json."
        ),
    )
    parser.add_argument("--include-feature-smoke", action="store_true", help="Include small feature-smoke benchmark jobs.")
    parser.add_argument("--no-feature-smoke", dest="include_feature_smoke", action="store_false")
    parser.set_defaults(include_feature_smoke=True)
    parser.add_argument("--include-validation-suite", action="store_true", help="Include validation_suite DF coverage + smoke.")
    parser.add_argument("--no-validation-suite", dest="include_validation_suite", action="store_false")
    parser.set_defaults(include_validation_suite=True)
    parser.add_argument(
        "--include-gpu-model-smoke",
        action="store_true",
        help="Include small smoke jobs that enable optional model candidates (tabpfn/xgb).",
    )
    args = parser.parse_args()

    num_pods = int(max(1, min(int(args.num_pods), DEFAULT_MAX_PODS)))
    plan_kind = str(getattr(args, "plan_kind", "validation1") or "validation1").strip().lower()

    # Plumb the optional GPU-model smoke flag via env var to keep the Job schema stable.
    if plan_kind == "legacy" and bool(getattr(args, "include_gpu_model_smoke", False)):
        os.environ["PODVAL_INCLUDE_GPU_MODEL_SMOKE"] = "1"

    raw_tag = str(getattr(args, "output_tag", "") or "").strip()
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_tag).strip("_.-")
    tag_suffix = f"_{safe_tag}" if safe_tag else ""

    val4_root_note = ""
    val5_root_note = ""
    val6_root_note = ""
    val7_root_note = ""
    val8_root_note = ""
    val9_root_note = ""
    val10_root_note = ""
    val11_root_note = ""
    val12_root_note = ""
    val13_root_note = ""
    val14_root_note = ""
    val15_root_note = ""
    val17_root_note = ""
    if plan_kind == "validation5":
        val4_root_raw_note = str(getattr(args, "val4_root", "") or "").strip()
        if val4_root_raw_note:
            val4_root_note = str(Path(val4_root_raw_note).expanduser().resolve())
        else:
            val4_root_note = str(
                (REPO_ROOT / "run_artifacts" / "validation-4" / "val4_6pods_merged").resolve()
            )
    if plan_kind == "validation6":
        val5_root_raw_note = str(getattr(args, "val5_root", "") or "").strip()
        if val5_root_raw_note:
            val5_root_note = str(Path(val5_root_raw_note).expanduser().resolve())
        else:
            val5_root_note = str(
                (REPO_ROOT / "run_artifacts" / "validation-5" / "val5_6pods_merged").resolve()
            )
    if plan_kind == "validation7":
        val6_root_raw_note = str(getattr(args, "val6_root", "") or "").strip()
        if val6_root_raw_note:
            val6_root_note = str(Path(val6_root_raw_note).expanduser().resolve())
        else:
            val6_root_note = str(
                (REPO_ROOT / "run_artifacts" / "validation-6" / "val6_finished_pull").resolve()
            )
    if plan_kind == "validation8":
        val7_root_raw_note = str(getattr(args, "val7_root", "") or "").strip()
        if val7_root_raw_note:
            val7_root_note = str(Path(val7_root_raw_note).expanduser().resolve())
        else:
            val7_root_note = str(
                (REPO_ROOT / "run_artifacts" / "validation-7" / "val7_6pods_merged").resolve()
            )
    if plan_kind == "validation9":
        val8_root_raw_note = str(getattr(args, "val8_root", "") or "").strip()
        if val8_root_raw_note:
            val8_root_note = str(Path(val8_root_raw_note).expanduser().resolve())
        else:
            val8_root_note = str(
                (REPO_ROOT / "run_artifacts" / "validation-8" / "val8_6pods_merged").resolve()
            )
    if plan_kind == "validation10":
        val9_root_raw_note = str(getattr(args, "val9_root", "") or "").strip()
        if val9_root_raw_note:
            val9_root_note = str(Path(val9_root_raw_note).expanduser().resolve())
        else:
            val9_root_note = str(
                (
                    REPO_ROOT
                    / "run_artifacts"
                    / "validation-9"
                    / "val9_6pods_live_pull_20260223_165714"
                ).resolve()
            )
    if plan_kind == "validation11":
        val10_root_raw_note = str(getattr(args, "val10_root", "") or "").strip()
        if val10_root_raw_note:
            val10_root_note = str(Path(val10_root_raw_note).expanduser().resolve())
        else:
            val10_root_note = str(
                (
                    REPO_ROOT
                    / "run_artifacts"
                    / "validation-10"
                    / "val10_finished_done_only_pull_20260227_111010"
                ).resolve()
            )
    if plan_kind == "validation12":
        val11_root_raw_note = str(getattr(args, "val11_root", "") or "").strip()
        if val11_root_raw_note:
            val11_root_note = str(Path(val11_root_raw_note).expanduser().resolve())
        else:
            val11_root_note = str(
                (
                    REPO_ROOT
                    / "run_artifacts"
                    / "validation-11-v2"
                    / "val11_4pods_live_20260227_183025"
                ).resolve()
            )
    if plan_kind in ("validation13", "validation13_clf_oracle"):
        val12_root_raw_note = str(getattr(args, "val12_root", "") or "").strip()
        if val12_root_raw_note:
            val12_root_note = str(Path(val12_root_raw_note).expanduser().resolve())
        else:
            val12_root_note = str(
                (
                    REPO_ROOT
                    / "run_artifacts"
                    / "validation-12"
                    / "val12_4pods_live_20260228_013532"
                ).resolve()
            )
        val11_root_raw_note = str(getattr(args, "val11_root", "") or "").strip()
        if val11_root_raw_note:
            val11_root_note = str(Path(val11_root_raw_note).expanduser().resolve())
        else:
            val11_root_note = str(
                (
                    REPO_ROOT
                    / "run_artifacts"
                    / "validation-11-v2"
                    / "val11_4pods_live_20260227_183025"
                ).resolve()
            )
        val10_root_raw_note = str(getattr(args, "val10_root", "") or "").strip()
        if val10_root_raw_note:
            val10_root_note = str(Path(val10_root_raw_note).expanduser().resolve())
        else:
            val10_root_note = str(
                (
                    REPO_ROOT
                    / "run_artifacts"
                    / "validation-10"
                    / "val10_finished_done_only_pull_20260227_111010"
                ).resolve()
            )
    if plan_kind in ("validation14", "validation14_activation_smoke"):
        val13_root_raw_note = str(getattr(args, "val13_root", "") or "").strip()
        if val13_root_raw_note:
            val13_root_note = str(Path(val13_root_raw_note).expanduser().resolve())
        else:
            val13_root_note = str(
                (
                    REPO_ROOT
                    / "run_artifacts"
                    / "validation-13"
                    / "val13_4hosts_live_20260228_133809"
                ).resolve()
            )
    if plan_kind == "validation15":
        val14_root_raw_note = str(getattr(args, "val14_root", "") or "").strip()
        if val14_root_raw_note:
            val14_root_note = str(Path(val14_root_raw_note).expanduser().resolve())
        else:
            val14_root_note = str(
                (
                    REPO_ROOT
                    / "run_artifacts"
                    / "validation-14"
                    / "val14_3hosts_live_20260301_223239"
                ).resolve()
            )
    if plan_kind in ("validation16", "validation17"):
        val15_root_raw_note = str(getattr(args, "val15_root", "") or "").strip()
        if val15_root_raw_note:
            val15_root_note = str(Path(val15_root_raw_note).expanduser().resolve())
        else:
            val15_root_note = str(
                (
                    REPO_ROOT
                    / "run_artifacts"
                    / "validation-15"
                    / "val15_rerun_3hosts_merged_20260305_124113"
                ).resolve()
            )
    if (
        plan_kind.startswith("validation18")
        or plan_kind in {"validation20_wave1", "validation20_ensemble"}
        or plan_kind == "validation19_classifiers"
    ):
        val17_root_raw_note = str(getattr(args, "val17_root", "") or "").strip()
        if val17_root_raw_note:
            val17_root_note = str(Path(val17_root_raw_note).expanduser().resolve())
        else:
            val17_root_note = str(
                (REPO_ROOT / "run_artifacts" / "validation-17").resolve()
            )

    plan_path = REPO_ROOT / "pod_validation" / f"plan_{num_pods}{tag_suffix}.json"
    shards_path = REPO_ROOT / "pod_validation" / f"shards_{num_pods}{tag_suffix}.json"
    split_md_path = REPO_ROOT / "pod_validation" / f"WORK_SPLIT_{num_pods}{tag_suffix}.md"

    if plan_kind == "legacy":
        jobs = build_jobs_legacy(
            include_feature_smoke=bool(args.include_feature_smoke),
            include_validation_suite=bool(args.include_validation_suite),
        )
    elif plan_kind == "validation4":
        jobs = build_jobs_validation4(dataset_shards=int(getattr(args, "dataset_shards", 6) or 6))
    elif plan_kind == "validation5":
        val4_root_raw = str(getattr(args, "val4_root", "") or "").strip()
        val4_root = Path(val4_root_raw).expanduser().resolve() if val4_root_raw else None
        jobs = build_jobs_validation5(
            dataset_shards=int(getattr(args, "dataset_shards", 6) or 6),
            val4_root=val4_root,
        )
    elif plan_kind == "validation6":
        val5_root_raw = str(getattr(args, "val5_root", "") or "").strip()
        val5_root = Path(val5_root_raw).expanduser().resolve() if val5_root_raw else None
        jobs = build_jobs_validation6(
            dataset_shards=int(getattr(args, "dataset_shards", 6) or 6),
            val5_root=val5_root,
        )
    elif plan_kind == "validation7":
        val6_root_raw = str(getattr(args, "val6_root", "") or "").strip()
        val6_root = Path(val6_root_raw).expanduser().resolve() if val6_root_raw else None
        jobs = build_jobs_validation7(
            dataset_shards=int(getattr(args, "dataset_shards", 6) or 6),
            val6_root=val6_root,
        )
    elif plan_kind == "validation8":
        val7_root_raw = str(getattr(args, "val7_root", "") or "").strip()
        val7_root = Path(val7_root_raw).expanduser().resolve() if val7_root_raw else None
        jobs = build_jobs_validation8(
            dataset_shards=int(getattr(args, "dataset_shards", 6) or 6),
            val7_root=val7_root,
        )
    elif plan_kind == "validation9":
        val8_root_raw = str(getattr(args, "val8_root", "") or "").strip()
        val8_root = Path(val8_root_raw).expanduser().resolve() if val8_root_raw else None
        jobs = build_jobs_validation9(
            dataset_shards=int(getattr(args, "dataset_shards", 6) or 6),
            val8_root=val8_root,
            runtime_profile=str(getattr(args, "val9_runtime_profile", "full") or "full"),
        )
    elif plan_kind == "validation10":
        val9_root_raw = str(getattr(args, "val9_root", "") or "").strip()
        val9_root = Path(val9_root_raw).expanduser().resolve() if val9_root_raw else None
        jobs = build_jobs_validation10(
            dataset_shards=int(getattr(args, "dataset_shards", 6) or 6),
            val9_root=val9_root,
        )
    elif plan_kind == "validation11":
        val10_root_raw = str(getattr(args, "val10_root", "") or "").strip()
        val10_root = Path(val10_root_raw).expanduser().resolve() if val10_root_raw else None
        jobs = build_jobs_validation11(
            dataset_shards=int(getattr(args, "dataset_shards", 6) or 6),
            val10_root=val10_root,
        )
    elif plan_kind == "validation12":
        val11_root_raw = str(getattr(args, "val11_root", "") or "").strip()
        val11_root = Path(val11_root_raw).expanduser().resolve() if val11_root_raw else None
        val10_root_raw = str(getattr(args, "val10_root", "") or "").strip()
        val10_root = Path(val10_root_raw).expanduser().resolve() if val10_root_raw else None
        jobs = build_jobs_validation12(
            dataset_shards=int(getattr(args, "dataset_shards", 6) or 6),
            val11_root=val11_root,
            val10_root=val10_root,
        )
    elif plan_kind == "validation13":
        val12_root_raw = str(getattr(args, "val12_root", "") or "").strip()
        val12_root = Path(val12_root_raw).expanduser().resolve() if val12_root_raw else None
        val11_root_raw = str(getattr(args, "val11_root", "") or "").strip()
        val11_root = Path(val11_root_raw).expanduser().resolve() if val11_root_raw else None
        jobs = build_jobs_validation13(
            dataset_shards=int(getattr(args, "dataset_shards", 8) or 8),
            val12_root=val12_root,
            val11_root=val11_root,
        )
    elif plan_kind == "validation13_clf_oracle":
        val12_root_raw = str(getattr(args, "val12_root", "") or "").strip()
        val12_root = Path(val12_root_raw).expanduser().resolve() if val12_root_raw else None
        jobs = build_jobs_validation13_clf_oracle(
            dataset_shards=int(getattr(args, "dataset_shards", 4) or 4),
            val12_root=val12_root,
        )
    elif plan_kind == "validation14":
        val13_root_raw = str(getattr(args, "val13_root", "") or "").strip()
        val13_root = Path(val13_root_raw).expanduser().resolve() if val13_root_raw else None
        jobs = build_jobs_validation14(
            dataset_shards=int(getattr(args, "dataset_shards", 6) or 6),
            val13_root=val13_root,
        )
    elif plan_kind == "validation14_activation_smoke":
        val13_root_raw = str(getattr(args, "val13_root", "") or "").strip()
        val13_root = Path(val13_root_raw).expanduser().resolve() if val13_root_raw else None
        jobs = build_jobs_validation14_activation_smoke(
            dataset_shards=int(getattr(args, "dataset_shards", 6) or 6),
            val13_root=val13_root,
        )
    elif plan_kind == "validation15":
        val14_root_raw = str(getattr(args, "val14_root", "") or "").strip()
        val14_root = Path(val14_root_raw).expanduser().resolve() if val14_root_raw else None
        jobs = build_jobs_validation15(
            dataset_shards=int(getattr(args, "dataset_shards", 6) or 6),
            val14_root=val14_root,
        )
    elif plan_kind == "validation16":
        val15_root_raw = str(getattr(args, "val15_root", "") or "").strip()
        val15_root = Path(val15_root_raw).expanduser().resolve() if val15_root_raw else None
        jobs = build_jobs_validation16(
            dataset_shards=int(getattr(args, "dataset_shards", 9) or 9),
            val15_root=val15_root,
        )
    elif plan_kind == "validation17":
        val15_root_raw = str(getattr(args, "val15_root", "") or "").strip()
        val15_root = Path(val15_root_raw).expanduser().resolve() if val15_root_raw else None
        jobs = build_jobs_validation17(
            dataset_shards=int(getattr(args, "dataset_shards", 9) or 9),
            val15_root=val15_root,
        )
    elif plan_kind == "validation18_anchors":
        val17_root_raw = str(getattr(args, "val17_root", "") or "").strip()
        val17_root = Path(val17_root_raw).expanduser().resolve() if val17_root_raw else None
        jobs = build_jobs_validation18_anchors(
            dataset_shards=int(getattr(args, "dataset_shards", 9) or 9),
            val17_root=val17_root,
        )
    elif plan_kind == "validation18_singletons":
        val17_root_raw = str(getattr(args, "val17_root", "") or "").strip()
        val17_root = Path(val17_root_raw).expanduser().resolve() if val17_root_raw else None
        jobs = build_jobs_validation18_singletons(
            dataset_shards=int(getattr(args, "dataset_shards", 9) or 9),
            val17_root=val17_root,
        )
    elif plan_kind == "validation18_mnpo":
        val17_root_raw = str(getattr(args, "val17_root", "") or "").strip()
        val17_root = Path(val17_root_raw).expanduser().resolve() if val17_root_raw else None
        jobs = build_jobs_validation18_mnpo(
            dataset_shards=int(getattr(args, "dataset_shards", 9) or 9),
            val17_root=val17_root,
        )
    elif plan_kind == "validation18_stage":
        val17_root_raw = str(getattr(args, "val17_root", "") or "").strip()
        val17_root = Path(val17_root_raw).expanduser().resolve() if val17_root_raw else None
        jobs = build_jobs_validation18_stage(
            dataset_shards=int(getattr(args, "dataset_shards", 7) or 7),
            val17_root=val17_root,
        )
    elif plan_kind == "validation18_classifiers":
        val17_root_raw = str(getattr(args, "val17_root", "") or "").strip()
        val17_root = Path(val17_root_raw).expanduser().resolve() if val17_root_raw else None
        jobs = build_jobs_validation18_classifiers(
            dataset_shards=int(getattr(args, "dataset_shards", 7) or 7),
            val17_root=val17_root,
        )
    elif plan_kind == "validation18_cls_oracle_wt":
        val17_root_raw = str(getattr(args, "val17_root", "") or "").strip()
        val17_root = Path(val17_root_raw).expanduser().resolve() if val17_root_raw else None
        jobs = build_jobs_validation18_cls_oracle_wt(
            dataset_shards=int(getattr(args, "dataset_shards", 7) or 7),
            val17_root=val17_root,
        )
    elif plan_kind == "validation19_classifiers":
        val17_root_raw = str(getattr(args, "val17_root", "") or "").strip()
        val17_root = Path(val17_root_raw).expanduser().resolve() if val17_root_raw else None
        jobs = build_jobs_validation19_classifiers(
            dataset_shards=int(getattr(args, "dataset_shards", 7) or 7),
            val17_root=val17_root,
        )
    elif plan_kind == "validation20_wave1":
        val17_root_raw = str(getattr(args, "val17_root", "") or "").strip()
        val17_root = Path(val17_root_raw).expanduser().resolve() if val17_root_raw else None
        jobs = build_jobs_validation20_wave1(
            dataset_shards=int(getattr(args, "dataset_shards", 8) or 8),
            val17_root=val17_root,
        )
    elif plan_kind == "validation20_ensemble":
        val17_root_raw = str(getattr(args, "val17_root", "") or "").strip()
        val17_root = Path(val17_root_raw).expanduser().resolve() if val17_root_raw else None
        jobs = build_jobs_validation20_ensemble(
            dataset_shards=int(getattr(args, "dataset_shards", 7) or 7),
            val17_root=val17_root,
        )
    elif plan_kind == "validation20_tabarena_w1":
        jobs = build_jobs_validation20_tabarena_w1(
            dataset_shards=int(getattr(args, "dataset_shards", 8) or 8),
        )
    elif plan_kind == "validation20_tabarena_w2":
        jobs = build_jobs_validation20_tabarena_w2(
            dataset_shards=int(getattr(args, "dataset_shards", 8) or 8),
        )
    elif plan_kind == "validation20_tabarena_w3":
        jobs = build_jobs_validation20_tabarena_w3(
            dataset_shards=int(getattr(args, "dataset_shards", 8) or 8),
        )
    else:
        jobs = build_jobs_validation1(dataset_shards=int(getattr(args, "dataset_shards", 8) or 8))
    job_count = len(jobs)
    ideal_pods = min(int(args.max_pods), job_count)

    if plan_kind == "validation5":
        shards = _balanced_shard_assign_validation5_pairs(jobs, num_pods)
    elif plan_kind == "validation6":
        shards = _balanced_shard_assign_validation6_pairs(jobs, num_pods)
    elif plan_kind == "validation7":
        shards = _balanced_shard_assign_validation7_pairs(jobs, num_pods)
    elif plan_kind == "validation8":
        shards = _balanced_shard_assign_validation8_pairs(jobs, num_pods)
    elif plan_kind == "validation9":
        shards = _balanced_shard_assign_validation9_pairs(jobs, num_pods)
    elif plan_kind == "validation10":
        shards = _balanced_shard_assign_validation10_pairs(jobs, num_pods)
    elif plan_kind == "validation11":
        shards = _balanced_shard_assign_validation11_triples(jobs, num_pods)
    elif plan_kind == "validation12":
        shards = _balanced_shard_assign_validation12_triples(jobs, num_pods)
    elif plan_kind == "validation13":
        shards = _balanced_shard_assign_validation13_bundles(jobs, num_pods)
    elif plan_kind == "validation13_clf_oracle":
        shards = _balanced_shard_assign_validation13_clf_oracle_bundles(jobs, num_pods)
    elif plan_kind == "validation14":
        shards = _balanced_shard_assign_validation14_bundles(jobs, num_pods)
    elif plan_kind == "validation14_activation_smoke":
        shards = _balanced_shard_assign_validation14_activation_bundles(jobs, num_pods)
    elif plan_kind == "validation15":
        shards = _balanced_shard_assign_validation15_bundles(jobs, num_pods)
    elif plan_kind == "validation16":
        shards = _balanced_shard_assign_validation16_bundles(jobs, num_pods)
    elif plan_kind == "validation17":
        shards = _balanced_shard_assign_validation17_bundles(jobs, num_pods)
    elif plan_kind == "validation18_anchors":
        shards = _balanced_shard_assign_validation18_anchors_bundles(jobs, num_pods)
    elif plan_kind == "validation18_singletons":
        shards = _balanced_shard_assign_validation18_singletons_bundles(jobs, num_pods)
    elif plan_kind == "validation18_mnpo":
        shards = _balanced_shard_assign_validation18_mnpo_bundles(jobs, num_pods)
    elif plan_kind == "validation18_stage":
        shards = _balanced_shard_assign_validation18_stage_bundles(jobs, num_pods)
    elif plan_kind == "validation18_classifiers":
        shards = _balanced_shard_assign_validation18_classifiers_bundles(jobs, num_pods)
    elif plan_kind == "validation18_cls_oracle_wt":
        shards = _balanced_shard_assign_validation18_cls_oracle_wt_bundles(jobs, num_pods)
    elif plan_kind == "validation19_classifiers":
        shards = _balanced_shard_assign_validation19_classifiers_bundles(jobs, num_pods)
    elif plan_kind == "validation20_wave1":
        shards = _balanced_shard_assign_validation20_wave1_bundles(jobs, num_pods)
    elif plan_kind == "validation20_ensemble":
        shards = _balanced_shard_assign_validation20_ensemble_bundles(jobs, num_pods)
    elif plan_kind in VALIDATION20_TABARENA_PLAN_KINDS:
        shards = _balanced_shard_assign_validation20_tabarena_bundles(jobs, num_pods)
    else:
        shards = _balanced_shard_assign(jobs, num_pods)
    jobs_by_id = {j.job_id: j for j in jobs}

    plan = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "job_count": job_count,
        "ideal_pods": ideal_pods,
        "num_shards": num_pods,
        "notes": {
            "max_pods_cap": int(args.max_pods),
            "plan_kind": plan_kind,
            "dataset_shards": int(getattr(args, "dataset_shards", 0) or 0),
            "output_tag": safe_tag,
            "val4_root": val4_root_note,
            "val5_root": val5_root_note,
            "val6_root": val6_root_note,
            "val7_root": val7_root_note,
            "val8_root": val8_root_note,
            "val9_root": val9_root_note,
            "val10_root": val10_root_note,
            "val11_root": val11_root_note,
            "val12_root": val12_root_note,
            "val13_root": val13_root_note,
            "val14_root": val14_root_note,
            "val15_root": val15_root_note,
            "val17_root": val17_root_note,
            "val9_runtime_profile": (
                str(getattr(args, "val9_runtime_profile", "full") or "full").strip().lower()
                if plan_kind == "validation9"
                else ""
            ),
            "excluded_method_sets": ["strict_plus_mrmr_auc_joint_l1"],
            "validation14_profile_manifest": (
                VALIDATION14_PROFILE_MANIFEST
                if plan_kind in ("validation14", "validation14_activation_smoke")
                else {}
            ),
            "validation15_profile_manifest": (
                VALIDATION15_PROFILE_MANIFEST
                if plan_kind == "validation15"
                else {}
            ),
            "validation16_profile_manifest": (
                VALIDATION16_PROFILE_MANIFEST
                if plan_kind == "validation16"
                else {}
            ),
            "validation17_profile_manifest": (
                VALIDATION17_PROFILE_MANIFEST
                if plan_kind == "validation17"
                else {}
            ),
            "validation18_profile_manifest": (
                VALIDATION18_ANCHORS_PROFILE_MANIFEST
                if plan_kind == "validation18_anchors"
                else VALIDATION18_MNPO_PROFILE_MANIFEST
                if plan_kind == "validation18_mnpo"
                else VALIDATION18_STAGE_PROFILE_MANIFEST
                if plan_kind == "validation18_stage"
                else VALIDATION18_CLASSIFIERS_PROFILE_MANIFEST
                if plan_kind == "validation18_classifiers"
                else VALIDATION18_CLS_ORACLE_WT_PROFILE_MANIFEST
                if plan_kind == "validation18_cls_oracle_wt"
                else {pid: {"variant": pid} for pid in VALIDATION18_SINGLETONS_PROFILE_IDS}
                if plan_kind == "validation18_singletons"
                else {}
            ),
            "validation18_host_policy": (
                {
                    "available_hosts": [*VAL18_CPU_HOSTS, *VAL18_GPU_HOSTS],
                    "cpu_hosts": list(VAL18_CPU_HOSTS),
                    "tabpfn_hosts": list(VAL18_GPU_HOSTS),
                    "arch_ml_reserved_for": "tabpfn",
                    "cpu_host_capacity_cores": dict(VAL18_CPU_HOST_CAPACITY_CORES),
                    "host_worker_targets": dict(VAL18_HOST_WORKER_TARGETS),
                    "recommended_host_assignment": _validation18_recommended_host_assignment(shards, jobs_by_id),
                }
                if str(plan_kind).startswith("validation18_")
                else {}
            ),
            "validation19_profile_manifest": (
                VALIDATION19_CLASSIFIERS_PROFILE_MANIFEST
                if plan_kind == "validation19_classifiers"
                else {}
            ),
            "validation19_pool_snapshots": (
                {
                    "added_classifiers": list(VAL19_ADDED_CLASSIFIERS),
                    "hdlss_extreme_old": list(VAL19_HDLSS_EXTREME_OLD),
                    "hdlss_extreme_new": list(VAL19_HDLSS_EXTREME_NEW),
                    "hdlss_moderate_old": list(VAL19_HDLSS_MODERATE_OLD),
                    "hdlss_moderate_new": list(VAL19_HDLSS_MODERATE_NEW),
                    "hdlss_moderate_cpu_old": list(VAL19_HDLSS_MODERATE_CPU_OLD),
                    "hdlss_moderate_cpu_new": list(VAL19_HDLSS_MODERATE_CPU_NEW),
                    "cpu_launch_excludes": ["tabpfn"],
                }
                if plan_kind == "validation19_classifiers"
                else {}
            ),
            "validation19_host_policy": (
                {
                    "available_hosts": list(VAL18_CPU_HOSTS),
                    "cpu_hosts": list(VAL18_CPU_HOSTS),
                    "cpu_only": True,
                    "tabpfn_hosts": [],
                    "cpu_host_capacity_cores": dict(VAL18_CPU_HOST_CAPACITY_CORES),
                    "host_worker_targets": dict(VAL19_HOST_WORKER_TARGETS),
                    "recommended_host_assignment": _validation19_recommended_host_assignment(shards, jobs_by_id),
                }
                if plan_kind == "validation19_classifiers"
                else {}
            ),
            "validation20_wave1_profile_manifest": (
                VALIDATION20_WAVE1_PROFILE_MANIFEST
                if plan_kind == "validation20_wave1"
                else {}
            ),
            "validation20_follow_on_profile_manifest": (
                {
                    "wave2_reserved": VALIDATION20_WAVE2_RESERVED_PROFILE_MANIFEST,
                    "wave3_reserved": VALIDATION20_WAVE3_RESERVED_PROFILE_MANIFEST,
                    "materialization_policy": (
                        "Reserved follow-on profiles are not emitted by validation20_wave1. "
                        "Instantiate them only after Wave 1 selects FLAML budgets, promoted "
                        "ensemble toggles, and composite-candidate settings."
                    ),
                }
                if plan_kind in {"validation20_wave1", "validation20_ensemble"}
                else {}
            ),
            "validation20_ensemble_profile_manifest": (
                VALIDATION20_ENSEMBLE_PROFILE_MANIFEST
                if plan_kind == "validation20_ensemble"
                else {}
            ),
            "validation20_tabarena_profile_manifest": (
                VALIDATION20_TABARENA_W1_PROFILE_MANIFEST
                if plan_kind == "validation20_tabarena_w1"
                else VALIDATION20_TABARENA_W2_PROFILE_MANIFEST
                if plan_kind == "validation20_tabarena_w2"
                else VALIDATION20_TABARENA_W3_PROFILE_MANIFEST
                if plan_kind == "validation20_tabarena_w3"
                else {}
            ),
            "validation20_pool_snapshots": (
                {
                    "core_pool": list(VAL20_CORE_CLASSIFIER_POOL),
                    "expanded_pool": list(VAL20_EXPANDED_CLASSIFIER_POOL),
                    "expanded_pool_no_tabpfn": list(VAL20_EXPANDED_CLASSIFIER_POOL_NO_TABPFN),
                    "custom_flaml_families": list(VAL20_CUSTOM_FLAML_FAMILIES),
                }
                if plan_kind in {"validation20_wave1", "validation20_ensemble"}
                else {}
            ),
            "validation20_host_policy": (
                {
                    "available_hosts": [*VAL20_CPU_HOSTS, *VAL20_GPU_HOSTS],
                    "cpu_hosts": list(VAL20_CPU_HOSTS),
                    "tabpfn_hosts": list(VAL20_GPU_HOSTS),
                    "cpu_host_capacity_cores": dict(VAL20_HOST_CAPACITY_CORES),
                    "host_worker_targets": dict(VAL20_HOST_WORKER_TARGETS),
                    "recommended_host_assignment": _validation20_recommended_host_assignment(shards, jobs_by_id),
                }
                if plan_kind in {"validation20_wave1", "validation20_ensemble"}
                else {}
            ),
            "validation20_tabarena_host_policy": (
                {
                    "available_hosts": list(VAL20_TABARENA_HOSTS),
                    "cpu_hosts": list(VAL20_TABARENA_HOSTS),
                    "cpu_only": True,
                    "cpu_host_capacity_cores": dict(VAL20_HOST_CAPACITY_CORES),
                    "host_worker_targets": dict(VAL20_HOST_WORKER_TARGETS),
                    "recommended_host_assignment": _validation20_tabarena_recommended_host_assignment(shards, jobs_by_id),
                }
                if plan_kind in VALIDATION20_TABARENA_PLAN_KINDS
                else {}
            ),
        },
        "jobs": [asdict(j) for j in jobs],
    }
    shard_doc = {
        "schema_version": 1,
        "generated_at": plan["generated_at"],
        "repo_root": str(REPO_ROOT),
        "num_shards": num_pods,
        "shards": {str(k): v for k, v in shards.items()},
        "validation18_recommended_host_assignment": (
            _validation18_recommended_host_assignment(shards, jobs_by_id)
            if str(plan_kind).startswith("validation18_")
            else {}
        ),
        "validation19_recommended_host_assignment": (
            _validation19_recommended_host_assignment(shards, jobs_by_id)
            if plan_kind == "validation19_classifiers"
            else {}
        ),
        "validation20_recommended_host_assignment": (
            _validation20_recommended_host_assignment(shards, jobs_by_id)
            if plan_kind in {"validation20_wave1", "validation20_ensemble"}
            else {}
        ),
        "validation20_tabarena_recommended_host_assignment": (
            _validation20_tabarena_recommended_host_assignment(shards, jobs_by_id)
            if plan_kind in VALIDATION20_TABARENA_PLAN_KINDS
            else {}
        ),
    }

    _write_json(plan_path, plan)
    _write_json(shards_path, shard_doc)
    _write_work_split_md(split_md_path, shards=shards, jobs_by_id=jobs_by_id)

    shard_totals = _shard_weight_totals(shards, jobs_by_id)
    min_w = min(shard_totals.values()) if shard_totals else 0.0
    max_w = max(shard_totals.values()) if shard_totals else 0.0
    mean_w = (sum(shard_totals.values()) / float(len(shard_totals))) if shard_totals else 0.0
    ratio = (max_w / min_w) if min_w > 0 else 0.0

    print(f"Wrote plan: {plan_path}")
    print(f"Wrote shards: {shards_path}")
    print(f"Wrote work split: {split_md_path}")
    print(f"job_count={job_count} ideal_pods={ideal_pods} (cap={int(args.max_pods)})")
    print(f"shard_weight_min={min_w:.1f} max={max_w:.1f} mean={mean_w:.1f} max/min={ratio:.3f}x")


if __name__ == "__main__":
    main()

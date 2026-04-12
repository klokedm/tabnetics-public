"""Validation catalog and dataset-set definitions."""

from __future__ import annotations

from typing import Dict, List

from tabnetics.datasets.registry import DATASET_REGISTRY, DatasetSpec as ValidationDatasetSpec


def _build_catalog() -> Dict[str, ValidationDatasetSpec]:
    # Validation-suite catalog is a view into the single source-of-truth registry.
    # Exclude synthetic benchmark-only datasets (these are generated in the benchmark runner).
    return {
        ds_id: spec
        for ds_id, spec in DATASET_REGISTRY.items()
        if str(getattr(spec, "source_kind", "")).strip().lower() != "synthetic"
    }


def _build_dataset_sets(catalog: Dict[str, ValidationDatasetSpec]) -> Dict[str, List[str]]:
    by_pipeline = {
        "fs": [k for k, v in catalog.items() if v.pipeline == "fs"],
        "df": [k for k, v in catalog.items() if v.pipeline == "df"],
        "integrated": [k for k, v in catalog.items() if v.pipeline == "integrated"],
    }

    sets: Dict[str, List[str]] = {
        "all": list(catalog.keys()),
        "smoke": ["leukemia_golub", "df_synthetic_parametric", "int_cdf_leukemia"],
        "fs_all": by_pipeline["fs"],
        "df_all": by_pipeline["df"],
        "integrated_all": by_pipeline["integrated"],
    }

    extended_only_ids = {
        k for k, v in catalog.items()
        if getattr(v, "extended_only", False)
    }
    sets["extended"] = list(catalog.keys())
    sets["core"] = [k for k in catalog if k not in extended_only_ids]

    for pipeline in ("fs", "df", "integrated"):
        for tier in ("easy", "medium", "hard", "very_hard"):
            key = f"{pipeline}_{tier}"
            sets[key] = [
                ds_id
                for ds_id, spec in catalog.items()
                if spec.pipeline == pipeline and spec.tier == tier
            ]

    return sets


CATALOG = _build_catalog()
DATASET_SETS = _build_dataset_sets(CATALOG)

__all__ = ["CATALOG", "DATASET_SETS", "ValidationDatasetSpec"]

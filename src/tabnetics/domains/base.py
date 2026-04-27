"""Shared dataset/domain context helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


_FACE_DOMAIN_FALLBACK_IDS = frozenset({"orlraws10p", "warp_pie10p", "pixraw10p"})


@dataclass(frozen=True)
class DatasetDomainContext:
    """Minimal domain metadata needed by domain-specialized helpers."""

    dataset_id: str
    display_name: str
    domain: str
    tier: str
    found_in_catalog: bool
    is_face_domain: bool

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def base_dataset_name(dataset_name: str) -> str:
    """Strip integrated-dataset suffixes and return the stable dataset id."""
    name = str(dataset_name or "").strip()
    if "__" in name:
        name = name.split("__", 1)[0]
    return name


def resolve_dataset_catalog_context(dataset_name: str) -> DatasetDomainContext:
    """Resolve catalog-backed domain metadata with a small fallback heuristic."""
    ds_id = base_dataset_name(dataset_name)
    if not ds_id:
        return DatasetDomainContext(
            dataset_id="",
            display_name="",
            domain="",
            tier="",
            found_in_catalog=False,
            is_face_domain=False,
        )

    try:
        from tabnetics.datasets.validation_catalog import CATALOG as _CATALOG
    except Exception:
        _CATALOG = {}

    spec = _CATALOG.get(ds_id)
    if spec is not None:
        display_name = str(getattr(spec, "display_name", ds_id) or ds_id)
        tier = str(getattr(spec, "tier", "") or "").strip().lower()
        params = dict(getattr(spec, "params", {}) or {})
        domain = str(params.get("domain", "") or "").strip().lower()
        is_face_domain = bool(domain == "face" or "face" in display_name.lower())
        return DatasetDomainContext(
            dataset_id=ds_id,
            display_name=display_name,
            domain=domain,
            tier=tier,
            found_in_catalog=True,
            is_face_domain=is_face_domain,
        )

    return DatasetDomainContext(
        dataset_id=ds_id,
        display_name=ds_id,
        domain="",
        tier="",
        found_in_catalog=False,
        is_face_domain=bool(ds_id in _FACE_DOMAIN_FALLBACK_IDS),
    )

"""Dataset registry, catalogs, loaders, and meta-features for validation-catalog and benchmark runs; when the HuggingFace bundle is configured, it acts as the authoritative operational mirror of the public upstream validation data sources."""

from .registry import *  # noqa: F401,F403
from .containers import *  # noqa: F401,F403
from .validation_catalog import CATALOG, DATASET_SETS
from .loaders import *  # noqa: F401,F403
from .meta_features import extract_meta_features

__all__ = [
    "CATALOG",
    "DATASET_SETS",
    "extract_meta_features",
]

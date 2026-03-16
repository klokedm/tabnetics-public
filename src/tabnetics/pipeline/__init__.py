"""Stable DF+FS pipeline surface with the promoted leakage-safe FS->DF workflow (``df_stage_position="after_fs"``), so distribution fitting operates on the feature space that survives selection rather than the full raw matrix."""

from .config import ClassificationConfig, DFFSConfig, DistributionFitterConfig
from .pipeline import DistributionFeatureSelectionPipeline

__all__ = [
    "ClassificationConfig",
    "DFFSConfig",
    "DistributionFitterConfig",
    "DistributionFeatureSelectionPipeline",
]

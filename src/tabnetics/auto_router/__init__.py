"""Packaged auto-router for tabnetics runtime calibration."""

from .runtime import (
    AUTO_ROUTER_ARTIFACT_VERSION,
    AutoRouterOutput,
    ScoreExpandedRouter,
    ScoreRouterConfig,
    apply_router_output,
    compute_dataset_descriptor,
    default_artifact_path,
    load_default_auto_router,
    predict_auto_router,
)

__all__ = [
    "AUTO_ROUTER_ARTIFACT_VERSION",
    "AutoRouterOutput",
    "ScoreExpandedRouter",
    "ScoreRouterConfig",
    "apply_router_output",
    "compute_dataset_descriptor",
    "default_artifact_path",
    "load_default_auto_router",
    "predict_auto_router",
]

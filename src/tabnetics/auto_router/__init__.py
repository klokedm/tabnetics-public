"""Packaged auto-router for tabnetics runtime calibration."""

from .runtime import (
    AUTO_ROUTER_ARTIFACT_VERSION,
    AutoRouterOutput,
    DescriptorOODGate,
    ScoreExpandedRouter,
    ScoreRouterConfig,
    apply_router_output,
    compute_dataset_descriptor,
    default_artifact_path,
    load_default_auto_router,
    predict_auto_router,
)
from .missingness import (
    MISSINGNESS_DESCRIPTOR_KEYS,
    MISSINGNESS_DESCRIPTOR_POLICY,
    compute_missingness_descriptors,
    missingness_descriptor_artifact_status,
    missingness_descriptor_model_input_enabled,
)
from .uncertainty import (
    CandidateUncertaintyPolicy,
    CrossFitRouterUncertaintyArtifact,
    RouterOutcomeRow,
    RouterUncertaintyError,
    fit_crossfit_router_uncertainty,
)

__all__ = [
    "AUTO_ROUTER_ARTIFACT_VERSION",
    "AutoRouterOutput",
    "DescriptorOODGate",
    "ScoreExpandedRouter",
    "ScoreRouterConfig",
    "apply_router_output",
    "compute_dataset_descriptor",
    "default_artifact_path",
    "load_default_auto_router",
    "predict_auto_router",
    "CandidateUncertaintyPolicy",
    "CrossFitRouterUncertaintyArtifact",
    "RouterOutcomeRow",
    "RouterUncertaintyError",
    "fit_crossfit_router_uncertainty",
    "MISSINGNESS_DESCRIPTOR_KEYS",
    "MISSINGNESS_DESCRIPTOR_POLICY",
    "compute_missingness_descriptors",
    "missingness_descriptor_artifact_status",
    "missingness_descriptor_model_input_enabled",
]

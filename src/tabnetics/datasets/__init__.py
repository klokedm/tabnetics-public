"""Dataset registry, catalogs, loaders, and meta-features for validation-catalog and benchmark runs; when the HuggingFace bundle is configured, it acts as the authoritative operational mirror of the public upstream validation data sources."""

from .registry import *  # noqa: F401,F403
from .containers import *  # noqa: F401,F403
from .validation_catalog import CATALOG, DATASET_SETS
from .loaders import *  # noqa: F401,F403
from .beyondarena import *  # noqa: F401,F403
from .beyondarena import __all__ as _beyondarena_all
from .meta_features import extract_meta_features
from .schema import (
    DatasetSchema,
    FeatureAnnotation,
    FeatureLineage,
    FeatureRole,
    FeatureSpec,
    InferenceSchemaCompatibilityReport,
    SchemaAlignmentMode,
    SchemaContractError,
    infer_dataset_schema,
)
from .external_validation import (
    EXTERNAL_COHORT_MANIFEST_SCHEMA_VERSION,
    DeclaredFeatureMapping,
    DeclaredLabelMapping,
    ExternalCohortContractError,
    ExternalCohortFamily,
    ExternalCohortManifest,
    ExternalEvidenceKind,
    ExternalMappingError,
    ExternalSourceResult,
    MappedExternalCohort,
    evaluate_external_source,
    map_external_cohort,
    skipped_external_source,
    summarize_external_sources,
)

__all__ = sorted(
    {
        "CATALOG",
        "DATASET_SETS",
        "DatasetSchema",
        "DeclaredFeatureMapping",
        "DeclaredLabelMapping",
        "EXTERNAL_COHORT_MANIFEST_SCHEMA_VERSION",
        "ExternalCohortContractError",
        "ExternalCohortFamily",
        "ExternalCohortManifest",
        "ExternalEvidenceKind",
        "ExternalMappingError",
        "ExternalSourceResult",
        "MappedExternalCohort",
        "FeatureAnnotation",
        "FeatureLineage",
        "FeatureRole",
        "FeatureSpec",
        "InferenceSchemaCompatibilityReport",
        "SchemaAlignmentMode",
        "SchemaContractError",
        "extract_meta_features",
        "infer_dataset_schema",
        "evaluate_external_source",
        "map_external_cohort",
        "skipped_external_source",
        "summarize_external_sources",
        *_beyondarena_all,
    }
)

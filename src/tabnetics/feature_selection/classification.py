"""Deprecated compatibility imports for the former classifier backend module.

Use :mod:`tabnetics.classification` for classifier backends and related public
types. This module retains only the supported legacy import names.
"""

import warnings as _warnings

from tabnetics.classification.backends import (
    REGIME_HDLSS_EXTREME,
    REGIME_HDLSS_MODERATE,
    REGIME_STANDARD,
    REGIME_POOLS,
    CLASSIFIER_COMPLEXITY_PRIOR,
    FLAML_NATIVE_BY_FAMILY,
    classify_regime,
    OracleCandidateStats,
    PLSDAClassifier,
    ClassifierBackend,
    SklearnBackend,
    FLAMLBackend,
    OptunaBackend,
    ClassifierOracle,
    MNPOClassifierBackend,
)

_warnings.warn(
    "tabnetics.feature_selection.classification is deprecated; "
    "use tabnetics.classification instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = (
    "REGIME_HDLSS_EXTREME",
    "REGIME_HDLSS_MODERATE",
    "REGIME_STANDARD",
    "REGIME_POOLS",
    "CLASSIFIER_COMPLEXITY_PRIOR",
    "FLAML_NATIVE_BY_FAMILY",
    "classify_regime",
    "OracleCandidateStats",
    "PLSDAClassifier",
    "ClassifierBackend",
    "SklearnBackend",
    "FLAMLBackend",
    "OptunaBackend",
    "ClassifierOracle",
    "MNPOClassifierBackend",
)

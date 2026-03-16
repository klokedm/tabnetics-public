"""Feature selection sub-package (Phase 2+3+6).

Re-exports:
- FeatureSelector class (Phase 6: moved to base.py)
- FeatureSelectionResult dataclass (Phase 6: moved to result.py)
- Method registry (Phase 2)
- Stability-base classes (Phase 2)
- Configuration dataclasses (Phase 3)
- Scoring cache (Phase 6: moved to scoring.py)
"""

from .base import FeatureSelector
from .result import FeatureSelectionResult
from .scoring import _FeatureScoreCache
from .config import FeatureSelectorConfig, OracleConfig
from .registry import METHOD_REGISTRY, MethodSpec, get_method_weights, get_experimental_keys
from .contracts import MethodContract, FeatureSelectorMethodContract, build_default_method_contracts

__all__ = [
    'FeatureSelector',
    'FeatureSelectionResult',
    'FeatureSelectorConfig',
    'OracleConfig',
    'METHOD_REGISTRY',
    'MethodSpec',
    'get_method_weights',
    'get_experimental_keys',
    'MethodContract',
    'FeatureSelectorMethodContract',
    'build_default_method_contracts',
]

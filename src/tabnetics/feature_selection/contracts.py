"""Method execution contracts for feature-selection methods.

T-P3-008 introduces an explicit contract layer so new methods can be
plugged in with predictable capabilities and runtime metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from .registry import METHOD_REGISTRY


class MethodContract(ABC):
    """Execution contract for one feature-selection method."""

    @property
    @abstractmethod
    def method_key(self) -> str:
        """Canonical registry key for the method."""

    @property
    @abstractmethod
    def requires_gpu(self) -> bool:
        """Whether method requires a CUDA-capable device."""

    @property
    @abstractmethod
    def estimated_runtime_class(self) -> str:
        """Runtime class: `fast`, `medium`, `slow`, or `gpu_required`."""

    @abstractmethod
    def supports_dataset(self, *, n_samples: int, n_features: int, n_classes: int) -> bool:
        """Return True when this method supports a dataset shape/regime."""

    @abstractmethod
    def compute(self, X: np.ndarray, y: np.ndarray, n_target_features: int):
        """Run the method and return ``(method_result, all_scores)``."""


@dataclass(frozen=True)
class FeatureSelectorMethodContract(MethodContract):
    """Thin adapter that routes the contract to an existing selector method."""

    key: str
    fn: Callable[[np.ndarray, np.ndarray, int], object]
    min_classes: int = 2
    binary_only: bool = False
    requires_multiclass: bool = False
    gpu_required: bool = False
    runtime_class: str = "medium"

    @property
    def method_key(self) -> str:
        return str(self.key)

    @property
    def requires_gpu(self) -> bool:
        return bool(self.gpu_required)

    @property
    def estimated_runtime_class(self) -> str:
        runtime = str(self.runtime_class).strip().lower()
        if runtime not in {"fast", "medium", "slow", "gpu_required"}:
            runtime = "medium"
        return runtime

    def supports_dataset(self, *, n_samples: int, n_features: int, n_classes: int) -> bool:
        if int(n_samples) < 2 or int(n_features) <= 0:
            return False
        classes = int(max(0, n_classes))
        if classes < int(max(2, self.min_classes)):
            return False
        if self.binary_only and classes != 2:
            return False
        if self.requires_multiclass and classes < 3:
            return False
        return True

    def compute(self, X: np.ndarray, y: np.ndarray, n_target_features: int):
        return self.fn(X, y, int(max(1, n_target_features)))


def build_default_method_contracts(selector) -> Dict[str, MethodContract]:
    """Build contracts for all registry-backed methods with callable handlers."""
    runtime_by_paradigm = {
        "filter": "fast",
        "embedded": "medium",
        "pairwise": "medium",
        "multiclass": "medium",
        "wrapper": "slow",
        "stability": "slow",
        "knockoff": "slow",
    }
    contracts: Dict[str, MethodContract] = {}
    for key, spec in METHOD_REGISTRY.items():
        if spec is None:
            continue
        if str(getattr(spec, "maturity", "stable")).strip().lower() == "deprecated":
            continue
        fn = getattr(selector, spec.fn_name, None)
        if fn is None:
            continue
        runtime = "gpu_required" if bool(getattr(spec, "requires_gpu", False)) else runtime_by_paradigm.get(
            str(getattr(spec, "paradigm", "filter")).strip().lower(),
            "medium",
        )
        contracts[key] = FeatureSelectorMethodContract(
            key=str(key),
            fn=fn,
            min_classes=int(getattr(spec, "min_classes", 2) or 2),
            binary_only=bool(getattr(spec, "binary_only", False)),
            requires_multiclass=bool(getattr(spec, "requires_multiclass", False)),
            gpu_required=bool(getattr(spec, "requires_gpu", False)),
            runtime_class=str(runtime),
        )
    return contracts

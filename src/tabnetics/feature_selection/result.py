"""FeatureSelectionResult dataclass — comprehensive output of the feature selection pipeline.

Extracted from ``tabnetics.feature_selection`` during Phase 6 module decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import numpy as np


@dataclass
class FeatureSelectionResult:
    """Comprehensive result object for feature selection process."""
    # Core results
    selected_feature_indices: np.ndarray
    selected_feature_votes: Dict[int, float]

    # Detailed feature information
    all_features_info: Dict[int, Dict[str, Any]]

    # Method-specific results
    method_results: Dict[str, Dict[str, Any]]

    # Preprocessing info
    eliminated_features: Dict[str, List[int]]  # reason -> list of indices

    # Optional reporting-only uncertainty diagnostics
    feature_importance_mean: Dict[int, float] = field(default_factory=dict)
    feature_importance_variance: Dict[int, float] = field(default_factory=dict)
    unstable_feature_indices: List[int] = field(default_factory=list)
    importance_uq: Dict[str, Any] = field(default_factory=dict)

    # Normalization info (if available from preprocessing)
    normalization_info: Optional[Dict[int, Dict[str, Any]]] = None

    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Emit a compact summary dictionary for per-run reporting.

        Returns a deterministic snapshot (modulo timestamps) suitable for
        JSON serialisation.  Includes ``schema_version`` per
        ArchitectureRefactor.md §14.3 OBS-1.
        """
        # Portfolio candidates = methods that produced results
        portfolio_candidates = sorted(self.method_results.keys()) if self.method_results else []

        # Oracle weights (from MNPO, if present in method_results)
        oracle_weights: Dict[str, float] = {}
        oracle_stability: Dict[str, Any] = {}
        copula_low_information: Dict[str, Any] = {}
        for method_key, minfo in (self.method_results or {}).items():
            if isinstance(minfo, dict) and "oracle_weight" in minfo:
                oracle_weights[method_key] = float(minfo["oracle_weight"])
            if isinstance(minfo, dict):
                pair_meta = minfo.get("oracle_pairwise_meta", {})
                if isinstance(pair_meta, dict) and isinstance(pair_meta.get("oracle_stability"), dict):
                    oracle_stability = dict(pair_meta.get("oracle_stability", {}))
                low_info = minfo.get("copula_low_information", {})
                if isinstance(low_info, dict):
                    copula_low_information = dict(low_info)

        # Selected feature indices (as plain list for JSON)
        if self.selected_feature_indices is not None and hasattr(self.selected_feature_indices, "tolist"):
            sel_indices = sorted(int(i) for i in self.selected_feature_indices.tolist())
        elif self.selected_feature_indices is not None:
            sel_indices = sorted(int(i) for i in self.selected_feature_indices)
        else:
            sel_indices = []

        return {
            "schema_version": "1.0",
            "fs_method_preset": self.config.get("method_set", "unknown"),
            "selection_strategy": self.config.get("selection_strategy", "unknown"),
            "portfolio_candidates": portfolio_candidates,
            "oracle_weights": oracle_weights,
            "oracle_stability": oracle_stability,
            "copula_low_information": copula_low_information,
            "n_features_selected": len(sel_indices),
            "selected_features": sel_indices,
            "n_methods_run": len(portfolio_candidates),
            "importance_uq": dict(self.importance_uq or {}),
            "n_unstable_features": len(set(int(i) for i in (self.unstable_feature_indices or []))),
            "unstable_features": sorted(set(int(i) for i in (self.unstable_feature_indices or []))),
            "eliminated_features_counts": {
                reason: len(indices)
                for reason, indices in (self.eliminated_features or {}).items()
            },
        }

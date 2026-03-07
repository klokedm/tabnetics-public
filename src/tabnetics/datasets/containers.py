"""First-class omics metadata containers (VAL12_Suggestions §3.1).

Lightweight data object for multi-omics datasets, inspired by AnnData /
SummarizedExperiment.  This module is opt-in and does not change any existing
pipeline interfaces.

Usage::

    from tabnetics.datasets.containers import OmicsDataset

    dataset = OmicsDataset(
        X=expression_matrix,
        y=labels,
        sample_meta=sample_df,
        feature_meta=feature_df,
        assay_type="microarray",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


@dataclass
class OmicsDataset:
    """Container for a single-assay or multi-omics expression dataset.

    Attributes
    ----------
    X : np.ndarray
        Expression matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Class label vector of shape ``(n_samples,)``.
    sample_meta : Optional[pd.DataFrame]
        Per-sample metadata (batch, cohort, age, ...) with shape
        ``(n_samples, k)``.  ``None`` when unavailable.
    feature_meta : Optional[pd.DataFrame]
        Per-feature metadata (gene symbol, chromosome, GO terms, ...) with
        shape ``(n_features, m)``.  ``None`` when unavailable.
    assay_type : str
        Free-text assay identifier (``"microarray"``, ``"rnaseq"``,
        ``"proteomics"``, ``"metabolomics"``, etc.).
    modality_membership : Optional[Dict[str, List[int]]]
        Feature indices per modality block for multi-omics integration.
        Keys are modality names, values are lists of column indices into *X*.
    batch_key : Optional[str]
        Column name in *sample_meta* that identifies batch membership.
    """

    X: np.ndarray
    y: np.ndarray
    sample_meta: Any = None  # Optional[pd.DataFrame]
    feature_meta: Any = None  # Optional[pd.DataFrame]
    assay_type: str = "unknown"
    modality_membership: Optional[Dict[str, List[int]]] = field(default=None)
    batch_key: Optional[str] = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y).ravel()
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {self.X.shape}")
        if self.y.size != self.X.shape[0]:
            raise ValueError(
                f"y length ({self.y.size}) != X row count ({self.X.shape[0]})"
            )
        self.assay_type = str(self.assay_type or "unknown").strip()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y).size)

    @property
    def class_counts(self) -> np.ndarray:
        _, counts = np.unique(self.y, return_counts=True)
        return counts

    @property
    def batch_labels(self) -> Optional[np.ndarray]:
        """Return batch labels from sample_meta or None."""
        if self.batch_key is None or self.sample_meta is None:
            return None
        if pd is not None and hasattr(self.sample_meta, "__getitem__"):
            try:
                return np.asarray(self.sample_meta[self.batch_key]).ravel()
            except (KeyError, TypeError):
                return None
        return None

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def subset_features(self, indices: Sequence[int]) -> "OmicsDataset":
        """Return a new dataset with only the specified feature columns."""
        idx = np.asarray(indices, dtype=int).ravel()
        new_X = self.X[:, idx]
        new_feature_meta = None
        if self.feature_meta is not None and pd is not None:
            try:
                new_feature_meta = self.feature_meta.iloc[idx].reset_index(drop=True)
            except Exception:
                new_feature_meta = None
        new_modality = None
        if self.modality_membership is not None:
            idx_set = set(int(i) for i in idx)
            old_to_new = {int(old): new_pos for new_pos, old in enumerate(idx)}
            new_modality = {}
            for name, old_indices in self.modality_membership.items():
                mapped = [old_to_new[i] for i in old_indices if i in idx_set]
                if mapped:
                    new_modality[name] = mapped
        return OmicsDataset(
            X=new_X,
            y=self.y.copy(),
            sample_meta=self.sample_meta.copy() if self.sample_meta is not None and hasattr(self.sample_meta, "copy") else self.sample_meta,
            feature_meta=new_feature_meta,
            assay_type=self.assay_type,
            modality_membership=new_modality,
            batch_key=self.batch_key,
        )

    def subset_samples(self, indices: Sequence[int]) -> "OmicsDataset":
        """Return a new dataset with only the specified sample rows."""
        idx = np.asarray(indices, dtype=int).ravel()
        new_X = self.X[idx]
        new_y = self.y[idx]
        new_sample_meta = None
        if self.sample_meta is not None and pd is not None:
            try:
                new_sample_meta = self.sample_meta.iloc[idx].reset_index(drop=True)
            except Exception:
                new_sample_meta = None
        return OmicsDataset(
            X=new_X,
            y=new_y,
            sample_meta=new_sample_meta,
            feature_meta=self.feature_meta.copy() if self.feature_meta is not None and hasattr(self.feature_meta, "copy") else self.feature_meta,
            assay_type=self.assay_type,
            modality_membership=dict(self.modality_membership) if self.modality_membership else None,
            batch_key=self.batch_key,
        )

    # ------------------------------------------------------------------
    # Train/test splitting
    # ------------------------------------------------------------------

    def train_test_split(
        self,
        test_size: float = 0.2,
        *,
        stratify: bool = True,
        random_state: int = 0,
    ) -> Tuple["OmicsDataset", "OmicsDataset"]:
        """Stratified train/test split returning two OmicsDataset objects."""
        from sklearn.model_selection import train_test_split as sk_split

        strat = self.y if stratify else None
        train_idx, test_idx = sk_split(
            np.arange(self.n_samples),
            test_size=float(test_size),
            stratify=strat,
            random_state=int(random_state),
        )
        return self.subset_samples(train_idx), self.subset_samples(test_idx)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_Xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract plain (X, y) arrays for pipeline consumption."""
        return self.X.copy(), self.y.copy()

    @classmethod
    def from_Xy(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        *,
        assay_type: str = "unknown",
        feature_names: Optional[Sequence[str]] = None,
        sample_ids: Optional[Sequence[str]] = None,
        batch_labels: Optional[np.ndarray] = None,
        batch_key: str = "batch",
    ) -> "OmicsDataset":
        """Create OmicsDataset from plain arrays with optional metadata."""
        sample_meta = None
        if pd is not None:
            meta_dict: Dict[str, Any] = {}
            if sample_ids is not None:
                meta_dict["sample_id"] = list(sample_ids)
            if batch_labels is not None:
                meta_dict[batch_key] = list(np.asarray(batch_labels).ravel())
            if meta_dict:
                sample_meta = pd.DataFrame(meta_dict)

        feature_meta = None
        if pd is not None and feature_names is not None:
            feature_meta = pd.DataFrame({"feature_name": list(feature_names)})

        return cls(
            X=np.asarray(X, dtype=float),
            y=np.asarray(y).ravel(),
            sample_meta=sample_meta,
            feature_meta=feature_meta,
            assay_type=str(assay_type),
            batch_key=batch_key if batch_labels is not None else None,
        )

    def __repr__(self) -> str:
        return (
            f"OmicsDataset(n_samples={self.n_samples}, n_features={self.n_features}, "
            f"n_classes={self.n_classes}, assay_type='{self.assay_type}')"
        )

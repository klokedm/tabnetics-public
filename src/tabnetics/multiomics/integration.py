"""DIABLO/MINT-style multi-omics integration (VAL12_Suggestions §4.1).

Provides supervised multi-block learning beyond simple concatenation:

- ``MultiBlockPLSDA``: supervised PLS-DA across multiple data blocks
  (modalities) with block-wise loadings and shared latent variables.
- ``MINTIntegrator``: study-aware integration that corrects for
  cohort/study effects while preserving biological signal.

Both classes operate on lists of (X_block, block_name) tuples and a
shared label vector *y*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg as sla


# ---------------------------------------------------------------------------
# MultiBlockPLSDA
# ---------------------------------------------------------------------------

@dataclass
class MultiBlockPLSDA:
    """Supervised multi-block PLS-DA (DIABLO-style).

    Fits a shared latent space across *K* data blocks by iteratively
    extracting latent components that maximise class discrimination
    in a one-vs-rest coding of *y*.

    Parameters
    ----------
    n_components : int
        Number of latent components to extract (default 2).
    max_iter : int
        Maximum NIPALS-style power iterations per component (default 500).
    tol : float
        Convergence tolerance on weight vector change (default 1e-6).
    scale_blocks : bool
        Whether to column-standardise each block before fitting
        (default True).
    """

    n_components: int = 2
    max_iter: int = 500
    tol: float = 1e-6
    scale_blocks: bool = True

    # Fitted state (populated after ``fit``)
    block_loadings_: List[np.ndarray] = field(default_factory=list, repr=False)
    block_weights_: List[np.ndarray] = field(default_factory=list, repr=False)
    block_scores_: List[np.ndarray] = field(default_factory=list, repr=False)
    super_scores_: Optional[np.ndarray] = field(default=None, repr=False)
    explained_variance_: Optional[np.ndarray] = field(default=None, repr=False)
    block_names_: List[str] = field(default_factory=list, repr=False)
    feature_importance_: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _block_means: List[np.ndarray] = field(default_factory=list, repr=False)
    _block_stds: List[np.ndarray] = field(default_factory=list, repr=False)
    _y_dummy: Optional[np.ndarray] = field(default=None, repr=False)
    _is_fitted: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        blocks: Sequence[Tuple[np.ndarray, str]],
        y: np.ndarray,
    ) -> "MultiBlockPLSDA":
        """Fit the multi-block PLS-DA model.

        Parameters
        ----------
        blocks : list of (X_block, block_name) tuples
            Each *X_block* is (n_samples, p_k).  All blocks must share
            the same sample axis length.
        y : (n_samples,) array of class labels.
        """
        if len(blocks) < 1:
            raise ValueError("At least one data block is required.")
        n_samples = int(blocks[0][0].shape[0])
        for X_b, name in blocks:
            if int(X_b.shape[0]) != n_samples:
                raise ValueError(
                    f"Block '{name}' has {X_b.shape[0]} samples but "
                    f"expected {n_samples}."
                )
        y_arr = np.asarray(y).ravel()
        if y_arr.size != n_samples:
            raise ValueError("y length does not match number of samples.")

        # One-hot dummy coding for Y.
        classes = np.unique(y_arr)
        Y = np.zeros((n_samples, classes.size), dtype=float)
        for ci, c in enumerate(classes):
            Y[y_arr == c, ci] = 1.0
        self._y_dummy = Y.copy()

        # Standardise blocks.
        Xbs: List[np.ndarray] = []
        self._block_means = []
        self._block_stds = []
        self.block_names_ = []
        for X_b, name in blocks:
            arr = np.asarray(X_b, dtype=float).copy()
            mu = np.mean(arr, axis=0)
            sd = np.std(arr, axis=0, ddof=1)
            sd = np.maximum(np.nan_to_num(sd, nan=1.0), 1e-8)
            if self.scale_blocks:
                arr = (arr - mu) / sd
            self._block_means.append(mu)
            self._block_stds.append(sd)
            self.block_names_.append(str(name))
            Xbs.append(arr)

        K = len(Xbs)
        n_comp = int(min(self.n_components, n_samples - 1))

        # Initialise storage.
        self.block_loadings_ = [np.zeros((Xbs[k].shape[1], n_comp)) for k in range(K)]
        self.block_weights_ = [np.zeros((Xbs[k].shape[1], n_comp)) for k in range(K)]
        self.block_scores_ = [np.zeros((n_samples, n_comp)) for k in range(K)]
        self.super_scores_ = np.zeros((n_samples, n_comp))
        self.explained_variance_ = np.zeros(n_comp)

        # Deflation copies.
        Xd = [X.copy() for X in Xbs]
        Yd = Y.copy()

        for comp in range(n_comp):
            # Initialise Y-space weight vector q (n_classes, 1).
            q = np.zeros((classes.size, 1), dtype=float)
            q[0, 0] = 1.0
            # Corresponding sample-space score: u = Y @ q  (n_samples, 1).
            u = Yd @ q
            u_norm = float(np.linalg.norm(u))
            if u_norm > 1e-12:
                u = u / u_norm

            for _it in range(self.max_iter):
                q_old = q.copy()
                # Block-level weight vectors.
                ws = []
                ts = []
                for k in range(K):
                    w_k = Xd[k].T @ u  # (p_k, 1)
                    w_norm = float(np.linalg.norm(w_k))
                    if w_norm > 1e-12:
                        w_k = w_k / w_norm
                    t_k = Xd[k] @ w_k  # (n, 1)
                    ws.append(w_k)
                    ts.append(t_k)

                # Super score: average block scores.
                t_super = np.mean(np.hstack(ts), axis=1, keepdims=True)
                t_norm = float(np.linalg.norm(t_super))
                if t_norm > 1e-12:
                    t_super = t_super / t_norm

                # Update Y-space weight q and sample score u.
                q = Yd.T @ t_super  # (n_classes, 1)
                q_norm = float(np.linalg.norm(q))
                if q_norm > 1e-12:
                    q = q / q_norm
                u = Yd @ q  # (n_samples, 1)
                u_norm = float(np.linalg.norm(u))
                if u_norm > 1e-12:
                    u = u / u_norm

                # Convergence check on Y-weight.
                diff = float(np.linalg.norm(q - q_old))
                if diff < self.tol:
                    break

            # Store component.
            self.super_scores_[:, comp] = t_super.ravel()
            for k in range(K):
                self.block_weights_[k][:, comp] = ws[k].ravel()
                t_k = Xd[k] @ ws[k]
                self.block_scores_[k][:, comp] = t_k.ravel()
                # Loading via regression.
                t_flat = t_k.ravel()
                denom = float(t_flat @ t_flat)
                if denom > 1e-12:
                    p_k = (Xd[k].T @ t_flat) / denom
                else:
                    p_k = np.zeros(Xd[k].shape[1])
                self.block_loadings_[k][:, comp] = p_k

            # Explained variance.
            var_comp = float(np.var(t_super.ravel()))
            self.explained_variance_[comp] = var_comp

            # Deflation.
            for k in range(K):
                t_k = self.block_scores_[k][:, comp : comp + 1]
                p_k = self.block_loadings_[k][:, comp : comp + 1]
                Xd[k] = Xd[k] - t_k @ p_k.T
            q = (Yd.T @ t_super)
            Yd = Yd - t_super @ q.T

        # Feature importance: sum of absolute weights across components.
        self.feature_importance_ = {}
        for k in range(K):
            imp = np.sum(np.abs(self.block_weights_[k]), axis=1)
            self.feature_importance_[self.block_names_[k]] = imp

        self._is_fitted = True
        return self

    def transform(
        self,
        blocks: Sequence[Tuple[np.ndarray, str]],
    ) -> np.ndarray:
        """Project new data into the shared latent space.

        Returns
        -------
        T : (n_samples, n_components) latent scores (averaged across blocks).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if len(blocks) != len(self.block_names_):
            raise ValueError(
                f"Expected {len(self.block_names_)} blocks, got {len(blocks)}."
            )
        n_samples = int(blocks[0][0].shape[0])
        K = len(blocks)
        n_comp = int(self.super_scores_.shape[1])
        all_scores = np.zeros((n_samples, n_comp, K))

        for k, (X_b, name) in enumerate(blocks):
            arr = np.asarray(X_b, dtype=float).copy()
            if self.scale_blocks:
                arr = (arr - self._block_means[k]) / self._block_stds[k]
            for comp in range(n_comp):
                w_k = self.block_weights_[k][:, comp]
                all_scores[:, comp, k] = arr @ w_k
                # Deflate.
                p_k = self.block_loadings_[k][:, comp]
                t_k = all_scores[:, comp : comp + 1, k]
                arr = arr - t_k @ p_k[None, :]

        return np.mean(all_scores, axis=2)

    def fit_transform(
        self,
        blocks: Sequence[Tuple[np.ndarray, str]],
        y: np.ndarray,
    ) -> np.ndarray:
        """Fit and return latent scores."""
        self.fit(blocks, y)
        return self.super_scores_.copy()

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Return per-block feature importance vectors."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        return dict(self.feature_importance_)


# ---------------------------------------------------------------------------
# MINTIntegrator
# ---------------------------------------------------------------------------

@dataclass
class MINTIntegrator:
    """MINT-style study-aware multi-block integrator.

    Fits per-study centering before applying multi-block PLS-DA so that
    cohort/batch effects are removed without leaking into the latent
    space.

    Parameters
    ----------
    n_components : int
        Number of latent components (default 2).
    scale_blocks : bool
        Column-standardise each block (default True).
    max_iter : int
        NIPALS iterations (default 500).
    tol : float
        Convergence tolerance (default 1e-6).
    """

    n_components: int = 2
    scale_blocks: bool = True
    max_iter: int = 500
    tol: float = 1e-6

    _study_means: Dict[str, List[np.ndarray]] = field(default_factory=dict, repr=False)
    _global_means: List[np.ndarray] = field(default_factory=list, repr=False)
    _pls: Optional[MultiBlockPLSDA] = field(default=None, repr=False)
    _is_fitted: bool = field(default=False, repr=False)

    def fit(
        self,
        blocks: Sequence[Tuple[np.ndarray, str]],
        y: np.ndarray,
        study_labels: np.ndarray,
    ) -> "MINTIntegrator":
        """Fit with per-study centering.

        Parameters
        ----------
        blocks : list of (X_block, block_name)
        y : class labels
        study_labels : (n_samples,) study/cohort labels
        """
        n_samples = int(blocks[0][0].shape[0])
        studies = np.asarray(study_labels).ravel()
        if studies.size != n_samples:
            raise ValueError("study_labels length mismatch.")

        unique_studies = np.unique(studies)
        K = len(blocks)

        # Compute global and study-level means for each block.
        self._global_means = []
        self._study_means = {}
        centred_blocks: List[Tuple[np.ndarray, str]] = []

        for k, (X_b, name) in enumerate(blocks):
            arr = np.asarray(X_b, dtype=float).copy()
            g_mean = np.mean(arr, axis=0)
            self._global_means.append(g_mean)

            for s in unique_studies:
                s_key = str(s)
                if s_key not in self._study_means:
                    self._study_means[s_key] = [None] * K
                mask = studies == s
                s_mean = np.mean(arr[mask], axis=0)
                self._study_means[s_key][k] = s_mean
                # Centre: remove study mean, add global mean.
                arr[mask] = arr[mask] - s_mean + g_mean

            centred_blocks.append((arr, name))

        self._pls = MultiBlockPLSDA(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            scale_blocks=self.scale_blocks,
        )
        self._pls.fit(centred_blocks, y)
        self._is_fitted = True
        return self

    def transform(
        self,
        blocks: Sequence[Tuple[np.ndarray, str]],
        study_labels: np.ndarray,
    ) -> np.ndarray:
        """Project new data with study-centering."""
        if not self._is_fitted or self._pls is None:
            raise RuntimeError("Model not fitted.")
        n_samples = int(blocks[0][0].shape[0])
        studies = np.asarray(study_labels).ravel()
        K = len(blocks)

        centred: List[Tuple[np.ndarray, str]] = []
        for k, (X_b, name) in enumerate(blocks):
            arr = np.asarray(X_b, dtype=float).copy()
            g_mean = self._global_means[k]
            for s in np.unique(studies):
                s_key = str(s)
                mask = studies == s
                if s_key in self._study_means and self._study_means[s_key][k] is not None:
                    s_mean = self._study_means[s_key][k]
                else:
                    # Unseen study: use its own mean (best-effort).
                    s_mean = np.mean(arr[mask], axis=0)
                arr[mask] = arr[mask] - s_mean + g_mean
            centred.append((arr, name))

        return self._pls.transform(centred)

    def fit_transform(
        self,
        blocks: Sequence[Tuple[np.ndarray, str]],
        y: np.ndarray,
        study_labels: np.ndarray,
    ) -> np.ndarray:
        """Fit and return latent scores."""
        self.fit(blocks, y, study_labels)
        return self._pls.super_scores_.copy()

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Return per-block feature importance from underlying PLS."""
        if not self._is_fitted or self._pls is None:
            raise RuntimeError("Model not fitted.")
        return self._pls.get_feature_importance()

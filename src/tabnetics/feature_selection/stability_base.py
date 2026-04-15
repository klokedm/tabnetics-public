"""Stability selection base class and variant implementations.

Extracts shared complementary-subsampling stability selection logic from
FeatureSelector into a template-method hierarchy.  Each variant preserves the
exact computational behaviour and RNG sequence of the original monolith
implementation.

Usage from FeatureSelector thin wrappers::

    def _stability_subsample_selection(self, X, y, n_target_features):
        runner = SubsampleStability(
            subsample_fraction=self.stability_subsample_fraction,
            selection_threshold=self.stability_selection_threshold,
            ...
        )
        return runner.run(
            X, y, n_target_features,
            self._prefilter_feature_pool,
            fit_score_fn=self._fit_and_score_fold,
        )
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

_SEED_MAX = 2**32  # numpy RandomState accepts [0, 2**32)


def _safe_seed(base: int, shift: int) -> int:
    """Derive a reproducible seed bounded to numpy's uint32 range."""
    return (base + shift) % _SEED_MAX


# ---------------------------------------------------------------------------
# Standalone helper (mirrors mnpo_core.normalize_vector_01)
# ---------------------------------------------------------------------------

def _normalize_vector_01(values):
    """Min-max normalize to [0,1] with safe fallback to 0.5 when constant."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if abs(max_v - min_v) < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - min_v) / (max_v - min_v)


# ===================================================================
# Base class
# ===================================================================

class StabilitySelectionBase(ABC):
    """Abstract base for complementary-subsampling stability selection.

    Template-method lifecycle:

    1. Pool setup  (``_setup_pool``)
    2. *Optional* pre-transform / early exit  (``_run_with_pool`` hook)
    3. Bootstrap configuration  (``_setup_bootstrap``)
    4. Variant-specific bootstrap body  (``_run_bootstrap`` – abstract)
    5. Feature selection from scored pool  (``_select_features`` hook)
    6. Result dict construction  (``_build_result``)

    Subclasses **must** implement ``_run_bootstrap``.
    They *may* override ``_run_with_pool`` (to inject pre-transforms or
    early-exit checks) and ``_select_features`` (for non-standard selection
    such as cluster-priority).
    """

    def __init__(
        self,
        *,
        subsample_fraction: float,
        selection_threshold: float,
        selection_threshold_method: str = "fixed",
        selection_target_pfer: float = 1.0,
        eats_exclusion_quantile: float = 0.90,
        eats_min_threshold: float = 0.45,
        n_bootstrap_iterations: int,
        random_state: int,
        problem_type: str,
        linear_svm_max_iter: int,
        mrmr_max_features: int,
        score_key: str = "stability_score",
        parallel_n_jobs: int = 1,
    ):
        self.subsample_fraction = subsample_fraction
        self.selection_threshold = selection_threshold
        method = str(selection_threshold_method or "fixed").strip().lower()
        self.selection_threshold_method = method if method in {"fixed", "eats", "cpss"} else "fixed"
        self.selection_target_pfer = float(max(1e-6, selection_target_pfer))
        self.eats_exclusion_quantile = float(np.clip(eats_exclusion_quantile, 0.50, 0.995))
        self.eats_min_threshold = float(np.clip(eats_min_threshold, 0.05, 0.95))
        self.n_bootstrap_iterations = n_bootstrap_iterations
        self.random_state = random_state
        self.problem_type = problem_type
        self.linear_svm_max_iter = linear_svm_max_iter
        self.mrmr_max_features = mrmr_max_features
        self.score_key = score_key
        self.parallel_n_jobs = int(parallel_n_jobs) if parallel_n_jobs is not None else 1

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_target_features: int,
        prefilter_fn: Callable,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[int, float]]:
        """Execute the stability selection variant.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,)
        n_target_features : int
        prefilter_fn : callable(X, y, max_features=int) -> ndarray of indices
        **kwargs : variant-specific (e.g. ``fit_score_fn`` for loss-guided)

        Returns
        -------
        results : dict  –  with ``selected_indices``, ``scores``, etc.
        all_scores : dict  –  ``{feature_index: float}`` for every feature
        """
        n_samples, n_features = X.shape
        if n_features == 0:
            return {}, {}

        pool_idx = self._setup_pool(X, y, n_target_features, prefilter_fn)
        if pool_idx.size == 0:
            return {}, {}
        X_pool = X[:, pool_idx]
        n_pool = X_pool.shape[1]

        return self._run_with_pool(
            X, X_pool, y, pool_idx, n_pool, n_samples, n_features,
            n_target_features, **kwargs,
        )

    def _run_with_pool(
        self, X, X_pool, y, pool_idx, n_pool, n_samples, n_features,
        n_target_features, **kwargs,
    ):
        """Core pipeline after pool setup.  Override for pre-transforms."""
        subsample_size, total_rounds, rng = self._setup_bootstrap(n_samples)

        bootstrap_result = self._run_bootstrap(
            X_pool, y, n_samples, subsample_size, total_rounds, rng,
            n_pool, **kwargs,
        )
        if bootstrap_result is None:
            return {}, {}

        selection_freq_pool, score_pool, extra_meta = bootstrap_result
        extra_meta.setdefault("pool_size", int(pool_idx.size))

        selected_local = self._select_features(
            score_pool, selection_freq_pool, n_target_features, extra_meta,
        )
        return self._build_result(
            selected_local, pool_idx, selection_freq_pool, score_pool,
            n_features, n_target_features, extra_meta,
        )

    # ------------------------------------------------------------------
    # Shared concrete helpers
    # ------------------------------------------------------------------

    def _setup_pool(self, X, y, n_target_features, prefilter_fn):
        """Compute compact candidate pool via *prefilter_fn*."""
        n_features = X.shape[1]
        pool_cap = int(
            min(
                n_features,
                max(
                    self.mrmr_max_features,
                    min(640, max(96, 8 * int(max(1, n_target_features)))),
                ),
            )
        )
        return prefilter_fn(X, y, max_features=pool_cap)

    def _setup_bootstrap(self, n_samples):
        """Compute subsample size, total rounds, and build deterministic RNG."""
        subsample_size = int(max(4, round(self.subsample_fraction * n_samples)))
        subsample_size = int(min(max(2, subsample_size), max(2, n_samples - 1)))
        total_rounds = int(max(2, self.n_bootstrap_iterations))
        rng = np.random.default_rng(self.random_state)
        return subsample_size, total_rounds, rng

    @staticmethod
    def _complementary_split(rng, n_samples, subsample_size):
        """Draw a complementary pair of subsamples."""
        subset = rng.choice(np.arange(n_samples), size=subsample_size, replace=False)
        complement_mask = np.ones(n_samples, dtype=bool)
        complement_mask[subset] = False
        complement = np.where(complement_mask)[0]
        return subset, complement

    def _fit_sparse_model(self, X_sub, y_sub, seed_shift):
        """Fit L1-penalised linear model, return ``|coefficients|``."""
        if self.problem_type == "classification":
            if np.unique(y_sub).size < 2:
                return None
            model = LinearSVC(
                penalty="l1",
                dual=False,
                C=0.18,
                random_state=_safe_seed(self.random_state, seed_shift),
                max_iter=self.linear_svm_max_iter,
            )
            model.fit(X_sub, y_sub)
            weights = np.abs(model.coef_)
            if weights.ndim == 2:
                weights = np.mean(weights, axis=0)
            return np.asarray(weights, dtype=float).ravel()
        model = Lasso(
            alpha=0.01,
            random_state=_safe_seed(self.random_state, seed_shift),
            max_iter=3000,
        )
        model.fit(X_sub, y_sub)
        return np.abs(np.asarray(model.coef_, dtype=float).ravel())

    def _standard_select(self, score_pool, freq_pool, n_target, threshold=None):
        """Threshold-then-rank selection.  Returns list of local pool indices."""
        if threshold is None:
            threshold = self.selection_threshold
        stable_idx = np.where(freq_pool >= threshold)[0]
        stable_ranked = (
            stable_idx[np.argsort(score_pool[stable_idx])[::-1]]
            if stable_idx.size > 0
            else np.array([], dtype=int)
        )
        ranked_all = np.argsort(score_pool)[::-1]

        selected_local: List[int] = []
        for idx in stable_ranked:
            selected_local.append(int(idx))
            if len(selected_local) >= n_target:
                break
        if len(selected_local) < n_target:
            for idx in ranked_all:
                if int(idx) not in selected_local:
                    selected_local.append(int(idx))
                if len(selected_local) >= n_target:
                    break
        return selected_local

    def _build_result(
        self,
        selected_local,
        pool_idx,
        selection_freq_pool,
        score_pool,
        n_features,
        n_target,
        extra_meta,
    ):
        """Map pool-space arrays to full feature space and assemble result dict."""
        selected_indices = pool_idx[np.array(selected_local[:n_target], dtype=int)]

        selection_freq = np.zeros(n_features, dtype=float)
        selection_freq[pool_idx] = selection_freq_pool

        score_full = np.zeros(n_features, dtype=float)
        score_full[pool_idx] = score_pool

        results: Dict[str, Any] = {
            "selected_indices": selected_indices,
            "scores": {int(idx): float(score_full[idx]) for idx in selected_indices},
            "selection_frequency": selection_freq,
            self.score_key: score_full,
        }

        # Map additional pool-space arrays to full feature space
        pool_arrays = extra_meta.pop("_pool_arrays", {})
        for key, arr in pool_arrays.items():
            full_arr = np.zeros(n_features, dtype=float)
            full_arr[pool_idx] = arr
            results[key] = full_arr

        results.update(extra_meta)

        all_scores = {i: float(score_full[i]) for i in range(n_features)}
        return results, all_scores

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _resolve_selection_threshold(self, freq_pool, extra_meta) -> float:
        """Resolve the per-run stability threshold."""
        fixed_threshold = float(np.clip(self.selection_threshold, 0.05, 0.99))
        method = str(getattr(self, "selection_threshold_method", "fixed") or "fixed").strip().lower()
        if method == "fixed":
            extra_meta["stability_threshold_method"] = "fixed"
            extra_meta["stability_selection_threshold_applied"] = float(fixed_threshold)
            return fixed_threshold

        if method == "eats":
            try:
                from .methods.stability_selection import eats_calibrate_threshold

                calibrated_threshold, eats_meta = eats_calibrate_threshold(
                    np.asarray(freq_pool, dtype=float),
                    null_scores=np.asarray(extra_meta.get("_eats_null_scores", np.array([])), dtype=float),
                    exclusion_quantile=float(getattr(self, "eats_exclusion_quantile", 0.90)),
                    min_threshold=float(getattr(self, "eats_min_threshold", 0.45)),
                    fallback_threshold=float(fixed_threshold),
                )
                extra_meta["stability_threshold_method"] = "eats"
                extra_meta["stability_selection_threshold_fixed"] = float(fixed_threshold)
                extra_meta["stability_selection_threshold_applied"] = float(calibrated_threshold)
                for key, value in dict(eats_meta or {}).items():
                    extra_meta[f"stability_{key}"] = value
                return float(np.clip(calibrated_threshold, 0.05, 0.99))
            except Exception as exc:
                extra_meta["stability_threshold_method"] = "eats_fallback_fixed"
                extra_meta["stability_selection_threshold_applied"] = float(fixed_threshold)
                extra_meta["stability_threshold_error"] = str(type(exc).__name__)
                return fixed_threshold

        if method == "cpss":
            try:
                from .methods.stability_selection import cpss_calibrate_threshold

                avg_selected_per_fit = extra_meta.get("stability_avg_selected_per_fit", None)
                calibrated_threshold, cpss_meta = cpss_calibrate_threshold(
                    np.asarray(freq_pool, dtype=float),
                    avg_selected_per_fit=(
                        None if avg_selected_per_fit is None else float(avg_selected_per_fit)
                    ),
                    n_features=int(extra_meta.get("pool_size", np.asarray(freq_pool, dtype=float).size) or 0),
                    target_pfer=float(getattr(self, "selection_target_pfer", 1.0) or 1.0),
                    fallback_threshold=float(fixed_threshold),
                )
                used_fallback = bool((cpss_meta or {}).get("cpss_used_fallback", False))
                extra_meta["stability_threshold_method"] = (
                    "cpss_fallback_fixed" if used_fallback else "cpss"
                )
                extra_meta["stability_selection_threshold_fixed"] = float(fixed_threshold)
                extra_meta["stability_selection_threshold_applied"] = float(calibrated_threshold)
                for key, value in dict(cpss_meta or {}).items():
                    extra_meta[f"stability_{key}"] = value
                return float(np.clip(calibrated_threshold, 0.05, 0.99))
            except Exception as exc:
                extra_meta["stability_threshold_method"] = "cpss_fallback_fixed"
                extra_meta["stability_selection_threshold_applied"] = float(fixed_threshold)
                extra_meta["stability_threshold_error"] = str(type(exc).__name__)
                return fixed_threshold

        extra_meta["stability_threshold_method"] = "fixed"
        extra_meta["stability_selection_threshold_applied"] = float(fixed_threshold)
        return fixed_threshold

    def _select_features(self, score_pool, freq_pool, n_target, extra_meta):
        """Override to customise selection (e.g. cluster-priority)."""
        threshold = self._resolve_selection_threshold(freq_pool, extra_meta)
        return self._standard_select(score_pool, freq_pool, n_target, threshold=threshold)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def _run_bootstrap(
        self, X_pool, y, n_samples, subsample_size, total_rounds, rng,
        n_pool, **kwargs,
    ):
        """Run the bootstrap iterations.

        Returns
        -------
        ``(selection_freq_pool, score_pool, extra_meta)`` or ``None``

        ``extra_meta`` may contain:
        * ``_pool_arrays`` – dict of ``{key: pool_array}`` mapped to full space
        * Other keys merged as-is into the result dict.
        """
        ...


# ===================================================================
# Variant 1: SubsampleStability
# ===================================================================

class SubsampleStability(StabilitySelectionBase):
    """Complementary-subsampling stability selection with sparse linear models.

    Optionally uses out-of-sample loss-guided validation to filter poor
    bootstrap fits.
    """

    def __init__(
        self,
        *,
        subsample_fraction: float,
        selection_threshold: float,
        selection_threshold_method: str = "fixed",
        selection_target_pfer: float = 1.0,
        eats_exclusion_quantile: float = 0.90,
        eats_min_threshold: float = 0.45,
        n_bootstrap_iterations: int,
        random_state: int,
        problem_type: str,
        linear_svm_max_iter: int,
        mrmr_max_features: int,
        use_loss_guided_validation: bool = False,
        validation_fraction: float = 0.25,
        validation_quantile: float = 0.40,
        validation_min_samples: int = 6,
        parallel_n_jobs: int = 1,
    ):
        super().__init__(
            subsample_fraction=subsample_fraction,
            selection_threshold=selection_threshold,
            selection_threshold_method=selection_threshold_method,
            selection_target_pfer=selection_target_pfer,
            eats_exclusion_quantile=eats_exclusion_quantile,
            eats_min_threshold=eats_min_threshold,
            n_bootstrap_iterations=n_bootstrap_iterations,
            random_state=random_state,
            problem_type=problem_type,
            linear_svm_max_iter=linear_svm_max_iter,
            mrmr_max_features=mrmr_max_features,
            score_key="stability_score",
            parallel_n_jobs=parallel_n_jobs,
        )
        self.use_loss_guided_validation = use_loss_guided_validation
        self.validation_fraction = validation_fraction
        self.validation_quantile = validation_quantile
        self.validation_min_samples = validation_min_samples

    # ---- bootstrap body ------------------------------------------------

    def _run_bootstrap(
        self, X_pool, y, n_samples, subsample_size, total_rounds, rng,
        n_pool, **kwargs,
    ):
        fit_score_fn = kwargs.get("fit_score_fn")

        coeff_records: List[np.ndarray] = []
        validation_scores: List[float] = []

        # Pre-generate all splits sequentially (preserves RNG sequence)
        splits: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
        for i in range(total_rounds):
            subset, complement = self._complementary_split(rng, n_samples, subsample_size)
            for j, idx_group in enumerate([subset, complement]):
                if idx_group.size < 3:
                    continue
                fit_idx, val_idx = self._split_fit_validation(
                    idx_group, y, rng, seed_shift=(2 * i + j),
                )
                splits.append((i, j, fit_idx, val_idx))

        def _process_split(args):
            i, j, fit_idx, val_idx = args
            if fit_idx.size < 3:
                return None
            coeffs = self._fit_sparse_model(
                X_pool[fit_idx], y[fit_idx], seed_shift=(2 * i + j),
            )
            if coeffs is None:
                return None
            coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)
            val_score = self._validation_score_for_coeffs(
                coeffs, fit_idx, val_idx, X_pool, y, fit_score_fn,
            )
            return (coeffs, val_score)

        _n_jobs = int(getattr(self, 'parallel_n_jobs', 1) or 1)
        if _n_jobs == -1:
            import os as _os
            _n_jobs = _os.cpu_count() or 1

        if _n_jobs > 1 and len(splits) > 1:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(_n_jobs, len(splits))) as executor:
                results = list(executor.map(_process_split, splits))
        else:
            results = [_process_split(s) for s in splits]

        for r in results:
            if r is not None:
                coeff_records.append(r[0])
                validation_scores.append(r[1])

        if not coeff_records:
            return None

        # --- loss-guided filtering ---
        total_fit_records = int(len(coeff_records))
        keep_mask = np.ones(total_fit_records, dtype=bool)
        validation_scores_arr = np.asarray(validation_scores, dtype=float)
        validation_threshold = float("nan")
        validation_fallback = False

        if self.use_loss_guided_validation:
            finite_mask = np.isfinite(validation_scores_arr)
            if np.any(finite_mask):
                validation_threshold = float(
                    np.quantile(validation_scores_arr[finite_mask], self.validation_quantile)
                )
                keep_mask = np.array(
                    [
                        (np.isfinite(score) and score >= validation_threshold)
                        for score in validation_scores_arr
                    ],
                    dtype=bool,
                )
                min_keep = int(max(1, round(0.25 * total_fit_records)))
                if int(np.sum(keep_mask)) < min_keep:
                    order = np.argsort(np.nan_to_num(validation_scores_arr, nan=-np.inf))[::-1]
                    keep_mask = np.zeros(total_fit_records, dtype=bool)
                    keep_mask[order[:min_keep]] = True
                    validation_fallback = True
            else:
                keep_mask = np.ones(total_fit_records, dtype=bool)
                validation_fallback = True

        # --- aggregate ---
        selection_counts = np.zeros(n_pool, dtype=float)
        coef_sum = np.zeros(n_pool, dtype=float)
        n_fits = 0
        selected_counts_per_fit: List[int] = []
        for coeffs, keep in zip(coeff_records, keep_mask):
            if not bool(keep):
                continue
            coef_sum += coeffs
            selected_mask = (coeffs > 1e-8).astype(float)
            selection_counts += selected_mask
            selected_counts_per_fit.append(int(np.sum(selected_mask)))
            n_fits += 1

        if n_fits == 0:
            for coeffs in coeff_records:
                coef_sum += coeffs
                selected_mask = (coeffs > 1e-8).astype(float)
                selection_counts += selected_mask
                selected_counts_per_fit.append(int(np.sum(selected_mask)))
            n_fits = int(len(coeff_records))
            validation_fallback = True

        selection_freq_pool = selection_counts / n_fits
        avg_coef_pool = coef_sum / n_fits
        coef_norm_pool = _normalize_vector_01(avg_coef_pool)
        stability_score_pool = 0.75 * selection_freq_pool + 0.25 * coef_norm_pool

        extra_meta: Dict[str, Any] = {
            "n_fits": int(n_fits),
            "n_fit_records": int(total_fit_records),
            "loss_guided_validation_enabled": bool(self.use_loss_guided_validation),
            "loss_guided_validation_fraction": float(self.validation_fraction),
            "loss_guided_validation_quantile": float(self.validation_quantile),
            "loss_guided_validation_threshold": float(validation_threshold),
            "loss_guided_validation_total_fits": int(total_fit_records),
            "loss_guided_validation_kept_fits": int(n_fits),
            "loss_guided_validation_fallback": bool(validation_fallback),
            "stability_avg_selected_per_fit": float(
                np.mean(selected_counts_per_fit) if selected_counts_per_fit else 0.0
            ),
        }
        return selection_freq_pool, stability_score_pool, extra_meta

    # ---- validation helpers -------------------------------------------

    def _split_fit_validation(self, idx_group, y, rng, seed_shift):
        """Split a half-sample into fit / validation for loss-guided filtering."""
        if (
            not self.use_loss_guided_validation
            or idx_group.size < max(6, self.validation_min_samples + 2)
        ):
            return idx_group, np.array([], dtype=int)

        n_val = int(round(self.validation_fraction * idx_group.size))
        n_val = int(max(self.validation_min_samples, n_val))
        n_val = int(min(max(1, n_val), idx_group.size - 2))
        if n_val <= 0:
            return idx_group, np.array([], dtype=int)

        y_group = y[idx_group]
        if self.problem_type == "classification":
            classes, counts = np.unique(y_group, return_counts=True)
            if classes.size >= 2 and np.min(counts) >= 2:
                try:
                    splitter = StratifiedShuffleSplit(
                        n_splits=1,
                        test_size=n_val,
                        random_state=_safe_seed(self.random_state, 101 + int(seed_shift)),
                    )
                    local_train, local_val = next(
                        splitter.split(np.zeros((idx_group.size, 1)), y_group)
                    )
                    return idx_group[local_train], idx_group[local_val]
                except Exception as exc:
                    pass

        perm = rng.permutation(idx_group.size)
        val_local = perm[:n_val]
        fit_local = perm[n_val:]
        if fit_local.size < 2:
            return idx_group, np.array([], dtype=int)
        return idx_group[fit_local], idx_group[val_local]

    def _validation_score_for_coeffs(
        self, coeffs, fit_idx, val_idx, X_pool, y, fit_score_fn,
    ):
        """Compute OOS validation score for a set of coefficients."""
        if not self.use_loss_guided_validation or val_idx.size < 2:
            return float("nan")
        selected_local = np.where(coeffs > 1e-8)[0]
        if selected_local.size == 0:
            selected_local = np.array([int(np.argmax(coeffs))], dtype=int)

        X_fit_sel = X_pool[fit_idx][:, selected_local]
        y_fit_sel = y[fit_idx]
        X_val_sel = X_pool[val_idx][:, selected_local]
        y_val_sel = y[val_idx]

        if self.problem_type == "classification":
            if np.unique(y_fit_sel).size < 2 or np.unique(y_val_sel).size < 1:
                return float("nan")
        try:
            score, _ = fit_score_fn(X_fit_sel, y_fit_sel, X_val_sel, y_val_sel)
            return float(score)
        except Exception as exc:
            return float("nan")


# ===================================================================
# Variant 2: TigressStability
# ===================================================================

class TigressStability(StabilitySelectionBase):
    """TIGRESS-style randomised stability-path selection.

    Randomised per-feature scaling + complementary subsampling over a sparse
    regularisation path, then path-integrated selection-frequency scoring.
    """

    def __init__(
        self,
        *,
        subsample_fraction: float,
        selection_threshold: float,
        selection_threshold_method: str = "fixed",
        selection_target_pfer: float = 1.0,
        eats_exclusion_quantile: float = 0.90,
        eats_min_threshold: float = 0.45,
        n_bootstrap_iterations: int,
        random_state: int,
        problem_type: str,
        linear_svm_max_iter: int,
        mrmr_max_features: int,
        ipss_min_c: float = 0.08,
        ipss_max_c: float = 1.20,
        ipss_path_grid_size: int = 7,
        parallel_n_jobs: int = 1,
    ):
        super().__init__(
            subsample_fraction=subsample_fraction,
            selection_threshold=selection_threshold,
            selection_threshold_method=selection_threshold_method,
            selection_target_pfer=selection_target_pfer,
            eats_exclusion_quantile=eats_exclusion_quantile,
            eats_min_threshold=eats_min_threshold,
            n_bootstrap_iterations=n_bootstrap_iterations,
            random_state=random_state,
            problem_type=problem_type,
            linear_svm_max_iter=linear_svm_max_iter,
            mrmr_max_features=mrmr_max_features,
            score_key="tigress_score",
            parallel_n_jobs=parallel_n_jobs,
        )
        self.ipss_min_c = ipss_min_c
        self.ipss_max_c = ipss_max_c
        self.ipss_path_grid_size = ipss_path_grid_size

    # ---- bootstrap body ------------------------------------------------

    def _run_bootstrap(
        self, X_pool, y, n_samples, subsample_size, total_rounds, rng,
        n_pool, **kwargs,
    ):
        if n_pool == 0:
            return None

        path_grid = np.logspace(
            np.log10(float(max(1e-3, self.ipss_min_c))),
            np.log10(float(max(self.ipss_min_c + 1e-3, self.ipss_max_c))),
            num=int(max(3, self.ipss_path_grid_size)),
        )
        n_path = int(len(path_grid))
        random_weight_low = 0.5

        selection_counts = np.zeros((n_path, n_pool), dtype=float)
        coef_sum = np.zeros((n_path, n_pool), dtype=float)
        path_fit_counts = np.zeros(n_path, dtype=float)

        # Pre-generate all splits + random scales sequentially (preserves RNG)
        splits: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(total_rounds):
            subset, complement = self._complementary_split(rng, n_samples, subsample_size)
            for j, idx_group in enumerate([subset, complement]):
                if idx_group.size < 3:
                    continue
                y_sub = y[idx_group]
                if self.problem_type == "classification" and np.unique(y_sub).size < 2:
                    continue
                random_scales = rng.uniform(random_weight_low, 1.0, size=n_pool)
                splits.append((i, j, idx_group, random_scales, y_sub))

        def _process_tigress_split(args):
            i, j, idx_group, random_scales, y_sub = args
            X_rand = X_pool[idx_group] * random_scales[None, :]
            local_results = []
            for p_idx, c_level in enumerate(path_grid):
                seed_shift = int((2 * i + j) * n_path + p_idx + 17)
                try:
                    if self.problem_type == "classification":
                        model = LinearSVC(
                            penalty="l1",
                            dual=False,
                            C=float(c_level),
                            random_state=_safe_seed(self.random_state, seed_shift),
                            max_iter=self.linear_svm_max_iter,
                        )
                        model.fit(X_rand, y_sub)
                        coeffs = np.abs(model.coef_)
                        if coeffs.ndim == 2:
                            coeffs = np.mean(coeffs, axis=0)
                        coeffs = np.asarray(coeffs, dtype=float).ravel()
                    else:
                        alpha = float(
                            np.clip(
                                1.0 / max(1e-8, float(c_level) * X_rand.shape[0]),
                                1e-4,
                                0.25,
                            )
                        )
                        model = Lasso(
                            alpha=alpha,
                            random_state=_safe_seed(self.random_state, seed_shift),
                            max_iter=3000,
                        )
                        model.fit(X_rand, y_sub)
                        coeffs = np.abs(np.asarray(model.coef_, dtype=float).ravel())
                except Exception as exc:
                    coeffs = None

                if coeffs is None or coeffs.size != n_pool:
                    continue
                coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)
                coeffs_orig = np.nan_to_num(
                    coeffs * random_scales,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                local_results.append((p_idx, coeffs_orig))
            return local_results

        _n_jobs = int(getattr(self, 'parallel_n_jobs', 1) or 1)
        if _n_jobs == -1:
            import os as _os
            _n_jobs = _os.cpu_count() or 1

        if _n_jobs > 1 and len(splits) > 1:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(_n_jobs, len(splits))) as executor:
                all_results = list(executor.map(_process_tigress_split, splits))
        else:
            all_results = [_process_tigress_split(s) for s in splits]

        selected_counts_per_fit: List[int] = []
        for local_results in all_results:
            for p_idx, coeffs_orig in local_results:
                coef_sum[p_idx] += coeffs_orig
                selected_mask = (coeffs_orig > 1e-8).astype(float)
                selection_counts[p_idx] += selected_mask
                path_fit_counts[p_idx] += 1.0
                selected_counts_per_fit.append(int(np.sum(selected_mask)))

        valid_mask = path_fit_counts > 0
        if not np.any(valid_mask):
            return None  # caller falls back to subsample

        # --- path-integrated aggregation ---
        selection_freq_path = np.zeros_like(selection_counts)
        selection_freq_path[valid_mask] = (
            selection_counts[valid_mask]
            / path_fit_counts[valid_mask][:, None]
        )
        mean_coef_path = np.zeros_like(coef_sum)
        mean_coef_path[valid_mask] = coef_sum[valid_mask] / path_fit_counts[valid_mask][:, None]

        path_x = np.linspace(0.0, 1.0, num=n_path)
        integrated_freq = np.trapezoid(selection_freq_path, x=path_x, axis=0)
        max_freq = np.max(selection_freq_path, axis=0)
        avg_coef = np.average(
            mean_coef_path, axis=0, weights=np.maximum(path_fit_counts, 1e-12),
        )
        coef_norm = _normalize_vector_01(avg_coef)
        tigress_score_pool = 0.65 * integrated_freq + 0.25 * max_freq + 0.10 * coef_norm

        stable_threshold = float(np.clip(self.selection_threshold, 0.05, 0.99))

        extra_meta: Dict[str, Any] = {
            "n_fits": int(np.sum(path_fit_counts)),
            "tigress_path_grid": [float(v) for v in np.asarray(path_grid, dtype=float).tolist()],
            "tigress_path_fit_counts": [int(v) for v in np.asarray(path_fit_counts, dtype=float).tolist()],
            "tigress_random_weight_low": float(random_weight_low),
            "tigress_selection_threshold": float(stable_threshold),
            "stability_avg_selected_per_fit": float(
                np.mean(selected_counts_per_fit) if selected_counts_per_fit else 0.0
            ),
            "_pool_arrays": {
                "tigress_max_frequency": max_freq,
            },
        }
        # Use integrated_freq as selection_frequency (matches original)
        return integrated_freq, tigress_score_pool, extra_meta

    def _select_features(self, score_pool, freq_pool, n_target, extra_meta):
        """Standard selection with TIGRESS-specific threshold clipping."""
        threshold = self._resolve_selection_threshold(freq_pool, extra_meta)
        extra_meta["tigress_selection_threshold"] = float(threshold)
        return self._standard_select(score_pool, freq_pool, n_target, threshold=threshold)


# ===================================================================
# Variant 3: DecorrelatedStability
# ===================================================================

class DecorrelatedStability(StabilitySelectionBase):
    """Decorrelated stability selection.

    Pre-whitens feature correlations, runs complementary-subsampling sparse
    selection in decorrelated space, then maps coefficients back to original
    feature coordinates.
    """

    def __init__(
        self,
        *,
        subsample_fraction: float,
        selection_threshold: float,
        selection_threshold_method: str = "fixed",
        selection_target_pfer: float = 1.0,
        eats_exclusion_quantile: float = 0.90,
        eats_min_threshold: float = 0.45,
        n_bootstrap_iterations: int,
        random_state: int,
        problem_type: str,
        linear_svm_max_iter: int,
        mrmr_max_features: int,
        decorrelated_stability_eps: float = 1e-3,
        decorrelated_stability_min_max_abs_corr: float = 0.0,
        parallel_n_jobs: int = 1,
    ):
        super().__init__(
            subsample_fraction=subsample_fraction,
            selection_threshold=selection_threshold,
            selection_threshold_method=selection_threshold_method,
            selection_target_pfer=selection_target_pfer,
            eats_exclusion_quantile=eats_exclusion_quantile,
            eats_min_threshold=eats_min_threshold,
            n_bootstrap_iterations=n_bootstrap_iterations,
            random_state=random_state,
            problem_type=problem_type,
            linear_svm_max_iter=linear_svm_max_iter,
            mrmr_max_features=mrmr_max_features,
            score_key="stability_score",
            parallel_n_jobs=parallel_n_jobs,
        )
        self.eps = decorrelated_stability_eps
        self.min_max_abs_corr = decorrelated_stability_min_max_abs_corr

    # ---- override _run_with_pool for pre-whitening + early exits ------

    def _run_with_pool(
        self, X, X_pool, y, pool_idx, n_pool, n_samples, n_features,
        n_target_features, **kwargs,
    ):
        if n_pool < 2:
            return {}, {}  # caller falls back to subsample

        # --- correlation analysis ---
        corr = np.corrcoef(X_pool, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)

        corr_abs = np.abs(corr)
        np.fill_diagonal(corr_abs, 0.0)
        max_abs_corr = float(np.max(corr_abs)) if corr_abs.size else 0.0
        min_gate = float(self.min_max_abs_corr)

        if min_gate > 0.0 and max_abs_corr < min_gate:
            results = {
                "selected_indices": np.array([], dtype=int),
                "scores": {},
                "method": "decorrelated_stability",
                "decorrelation_gated": True,
                "decorrelation_gate_reason": "max_abs_corr_below_threshold",
                "decorrelation_max_abs_corr": float(max_abs_corr),
                "decorrelation_min_max_abs_corr": float(min_gate),
                "pool_size": int(pool_idx.size),
            }
            return results, {}

        # --- whitening transform ---
        eps = float(self.eps)
        corr_reg = corr + eps * np.eye(n_pool, dtype=float)

        try:
            eigvals, eigvecs = np.linalg.eigh(corr_reg)
        except np.linalg.LinAlgError:
            return {}, {}  # caller falls back to subsample

        eigvals = np.clip(np.asarray(eigvals, dtype=float), 1e-8, None)
        inv_sqrt = (eigvecs / np.sqrt(eigvals)[None, :]) @ eigvecs.T
        X_decor = np.asarray(X_pool @ inv_sqrt, dtype=float)
        cond_number = float(np.max(eigvals) / max(np.min(eigvals), 1e-12))
        eff_rank = int(np.sum(eigvals > 1e-6))

        # --- standard bootstrap in decorrelated space ---
        subsample_size, total_rounds, rng = self._setup_bootstrap(n_samples)
        bootstrap_result = self._run_bootstrap(
            X_decor, y, n_samples, subsample_size, total_rounds, rng,
            n_pool, inv_sqrt=inv_sqrt,
        )
        if bootstrap_result is None:
            return {}, {}

        selection_freq_pool, score_pool, extra_meta = bootstrap_result
        extra_meta.setdefault("pool_size", int(pool_idx.size))
        extra_meta.update({
            "decorrelation_condition_number": float(cond_number),
            "decorrelation_rank": int(eff_rank),
            "decorrelation_eps": float(eps),
            "decorrelation_gated": False,
            "decorrelation_gate_reason": "",
            "decorrelation_max_abs_corr": float(max_abs_corr),
            "decorrelation_min_max_abs_corr": float(min_gate),
        })

        selected_local = self._select_features(
            score_pool, selection_freq_pool, n_target_features, extra_meta,
        )
        return self._build_result(
            selected_local, pool_idx, selection_freq_pool, score_pool,
            n_features, n_target_features, extra_meta,
        )

    # ---- bootstrap body: fit in whitened space, map back ---------------

    def _run_bootstrap(
        self, X_pool, y, n_samples, subsample_size, total_rounds, rng,
        n_pool, **kwargs,
    ):
        inv_sqrt = kwargs.get("inv_sqrt")
        if inv_sqrt is None:
            raise ValueError("DecorrelatedStability requires inv_sqrt kwarg")

        selection_counts = np.zeros(n_pool, dtype=float)
        coef_sum = np.zeros(n_pool, dtype=float)
        n_fits = 0

        # Pre-generate all splits sequentially (preserves RNG sequence)
        splits: List[Tuple[int, int, np.ndarray]] = []
        for i in range(total_rounds):
            subset, complement = self._complementary_split(rng, n_samples, subsample_size)
            for j, idx_group in enumerate([subset, complement]):
                if idx_group.size < 3:
                    continue
                splits.append((i, j, idx_group))

        def _process_decor_split(args):
            i, j, idx_group = args
            coeffs_decor = self._fit_sparse_model(
                X_pool[idx_group], y[idx_group], seed_shift=(2 * i + j),
            )
            if coeffs_decor is None:
                return None
            coeffs_decor = np.nan_to_num(coeffs_decor, nan=0.0, posinf=0.0, neginf=0.0)
            coeffs_orig = np.abs(inv_sqrt @ coeffs_decor)
            coeffs_orig = np.nan_to_num(coeffs_orig, nan=0.0, posinf=0.0, neginf=0.0)
            return coeffs_orig

        _n_jobs = int(getattr(self, 'parallel_n_jobs', 1) or 1)
        if _n_jobs == -1:
            import os as _os
            _n_jobs = _os.cpu_count() or 1

        if _n_jobs > 1 and len(splits) > 1:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(_n_jobs, len(splits))) as executor:
                results = list(executor.map(_process_decor_split, splits))
        else:
            results = [_process_decor_split(s) for s in splits]

        selected_counts_per_fit: List[int] = []
        for coeffs_orig in results:
            if coeffs_orig is not None:
                coef_sum += coeffs_orig
                selected_mask = (coeffs_orig > 1e-8).astype(float)
                selection_counts += selected_mask
                selected_counts_per_fit.append(int(np.sum(selected_mask)))
                n_fits += 1

        if n_fits == 0:
            return None

        selection_freq_pool = selection_counts / n_fits
        avg_coef_pool = coef_sum / n_fits
        coef_norm_pool = _normalize_vector_01(avg_coef_pool)
        stability_score_pool = 0.75 * selection_freq_pool + 0.25 * coef_norm_pool

        extra_meta: Dict[str, Any] = {
            "n_fits": int(n_fits),
            "stability_avg_selected_per_fit": float(
                np.mean(selected_counts_per_fit) if selected_counts_per_fit else 0.0
            ),
        }
        return selection_freq_pool, stability_score_pool, extra_meta


# ===================================================================
# Variant 4: ClusterStability
# ===================================================================

class ClusterStability(StabilitySelectionBase):
    """Cluster-aware stability selection for correlated HDLSS features.

    Features are prioritised by cluster-level stability and within-cluster
    support.  Uses correlation-based connected-component clustering.
    """

    def __init__(
        self,
        *,
        subsample_fraction: float,
        selection_threshold: float,
        selection_threshold_method: str = "fixed",
        selection_target_pfer: float = 1.0,
        eats_exclusion_quantile: float = 0.90,
        eats_min_threshold: float = 0.45,
        n_bootstrap_iterations: int,
        random_state: int,
        problem_type: str,
        linear_svm_max_iter: int,
        mrmr_max_features: int,
        cluster_stability_corr_threshold: float = 0.85,
        cluster_stability_max_per_cluster: int = 2,
        cluster_stability_min_cluster_freq: float = 0.55,
        parallel_n_jobs: int = 1,
    ):
        super().__init__(
            subsample_fraction=subsample_fraction,
            selection_threshold=selection_threshold,
            selection_threshold_method=selection_threshold_method,
            selection_target_pfer=selection_target_pfer,
            eats_exclusion_quantile=eats_exclusion_quantile,
            eats_min_threshold=eats_min_threshold,
            n_bootstrap_iterations=n_bootstrap_iterations,
            random_state=random_state,
            problem_type=problem_type,
            linear_svm_max_iter=linear_svm_max_iter,
            mrmr_max_features=mrmr_max_features,
            score_key="stability_score",
            parallel_n_jobs=parallel_n_jobs,
        )
        self.corr_threshold = cluster_stability_corr_threshold
        self.max_per_cluster = cluster_stability_max_per_cluster
        self.min_cluster_freq = cluster_stability_min_cluster_freq

    # ---- override _run_with_pool to build clusters then bootstrap ------

    def _run_with_pool(
        self, X, X_pool, y, pool_idx, n_pool, n_samples, n_features,
        n_target_features, **kwargs,
    ):
        clusters = self._build_correlation_clusters(X_pool)
        if not clusters:
            return {}, {}  # caller falls back to subsample

        cluster_of = np.zeros(n_pool, dtype=int)
        for cid, members in enumerate(clusters):
            for m in members:
                cluster_of[int(m)] = int(cid)

        subsample_size, total_rounds, rng = self._setup_bootstrap(n_samples)
        bootstrap_result = self._run_bootstrap(
            X_pool, y, n_samples, subsample_size, total_rounds, rng,
            n_pool, cluster_of=cluster_of, n_clusters=len(clusters),
        )
        if bootstrap_result is None:
            return {}, {}

        selection_freq_pool, score_pool, extra_meta = bootstrap_result
        extra_meta.setdefault("pool_size", int(pool_idx.size))
        extra_meta.update({
            "n_clusters": int(len(clusters)),
            "cluster_sizes": [int(len(c)) for c in clusters],
        })
        # Store cluster data for selection
        extra_meta["_cluster_of"] = cluster_of
        extra_meta["_clusters"] = clusters
        extra_meta["_cluster_freq"] = extra_meta.pop("_raw_cluster_freq")

        selected_local = self._select_features(
            score_pool, selection_freq_pool, n_target_features, extra_meta,
        )

        # Clean internal keys before building result
        cluster_freq = extra_meta.pop("_cluster_freq")
        extra_meta.pop("_cluster_of", None)
        extra_meta.pop("_clusters", None)
        extra_meta["cluster_frequency"] = [float(v) for v in cluster_freq.tolist()]

        return self._build_result(
            selected_local, pool_idx, selection_freq_pool, score_pool,
            n_features, n_target_features, extra_meta,
        )

    # ---- cluster building ----------------------------------------------

    def _build_correlation_clusters(self, X_pool):
        """Build connected components using absolute-correlation thresholding."""
        n_pool = X_pool.shape[1]
        if n_pool == 0:
            return []
        if n_pool == 1:
            return [[0]]

        corr = np.corrcoef(X_pool, rowvar=False)
        corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
        adj = corr >= self.corr_threshold
        np.fill_diagonal(adj, True)

        visited = np.zeros(n_pool, dtype=bool)
        clusters: List[List[int]] = []
        for start in range(n_pool):
            if visited[start]:
                continue
            stack = [int(start)]
            visited[start] = True
            comp: List[int] = []
            while stack:
                node = int(stack.pop())
                comp.append(node)
                neighbors = np.where(adj[node])[0]
                for nb in neighbors:
                    nb = int(nb)
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            clusters.append(sorted(comp))

        clusters.sort(key=len, reverse=True)
        return clusters

    # ---- bootstrap body with cluster tracking --------------------------

    def _run_bootstrap(
        self, X_pool, y, n_samples, subsample_size, total_rounds, rng,
        n_pool, **kwargs,
    ):
        cluster_of = kwargs["cluster_of"]
        n_clusters = kwargs["n_clusters"]

        selection_counts = np.zeros(n_pool, dtype=float)
        coef_sum = np.zeros(n_pool, dtype=float)
        cluster_hits = np.zeros(n_clusters, dtype=float)
        n_fits = 0

        # Pre-generate all splits sequentially (preserves RNG sequence)
        splits: List[Tuple[int, int, np.ndarray]] = []
        for i in range(total_rounds):
            subset, complement = self._complementary_split(rng, n_samples, subsample_size)
            for j, idx_group in enumerate([subset, complement]):
                if idx_group.size < 3:
                    continue
                splits.append((i, j, idx_group))

        def _process_cluster_split(args):
            i, j, idx_group = args
            coeffs = self._fit_sparse_model(
                X_pool[idx_group], y[idx_group], seed_shift=(2 * i + j),
            )
            if coeffs is None:
                return None
            coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)
            return coeffs

        _n_jobs = int(getattr(self, 'parallel_n_jobs', 1) or 1)
        if _n_jobs == -1:
            import os as _os
            _n_jobs = _os.cpu_count() or 1

        if _n_jobs > 1 and len(splits) > 1:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(_n_jobs, len(splits))) as executor:
                results = list(executor.map(_process_cluster_split, splits))
        else:
            results = [_process_cluster_split(s) for s in splits]

        selected_counts_per_fit: List[int] = []
        for coeffs in results:
            if coeffs is not None:
                selected_local = np.where(coeffs > 1e-8)[0]
                if selected_local.size > 0:
                    cluster_hits[np.unique(cluster_of[selected_local])] += 1.0
                coef_sum += coeffs
                selected_mask = (coeffs > 1e-8).astype(float)
                selection_counts += selected_mask
                selected_counts_per_fit.append(int(np.sum(selected_mask)))
                n_fits += 1

        if n_fits == 0:
            return None

        selection_freq_pool = selection_counts / n_fits
        avg_coef_pool = coef_sum / n_fits
        coef_norm_pool = _normalize_vector_01(avg_coef_pool)
        cluster_freq = cluster_hits / n_fits
        cluster_signal_pool = (
            cluster_freq[cluster_of] if cluster_freq.size else np.zeros(n_pool, dtype=float)
        )

        stability_score_pool = (
            0.55 * selection_freq_pool
            + 0.20 * coef_norm_pool
            + 0.25 * cluster_signal_pool
        )

        extra_meta: Dict[str, Any] = {
            "n_fits": int(n_fits),
            "_raw_cluster_freq": cluster_freq,
            "stability_avg_selected_per_fit": float(
                np.mean(selected_counts_per_fit) if selected_counts_per_fit else 0.0
            ),
        }
        return selection_freq_pool, stability_score_pool, extra_meta

    # ---- cluster-priority selection ------------------------------------

    def _select_features(self, score_pool, freq_pool, n_target, extra_meta):
        """Cluster-priority round-robin then fill by score."""
        threshold = self._resolve_selection_threshold(freq_pool, extra_meta)
        extra_meta["cluster_selection_threshold"] = float(threshold)
        cluster_of = extra_meta["_cluster_of"]
        clusters = extra_meta["_clusters"]
        cluster_freq = extra_meta["_cluster_freq"]

        cluster_order = np.argsort(cluster_freq)[::-1]
        selected_local: List[int] = []
        selected_set: set = set()
        per_cluster_counts = np.zeros(len(clusters), dtype=int)

        # First pass: one best feature per high-frequency cluster
        for cid in cluster_order:
            cid = int(cid)
            if cluster_freq[cid] < self.min_cluster_freq:
                continue
            members = [int(m) for m in clusters[cid] if float(freq_pool[int(m)]) >= float(threshold)]
            if not members:
                members = [int(m) for m in clusters[cid]]
            ranked = sorted(
                members, key=lambda idx: score_pool[int(idx)], reverse=True,
            )
            if not ranked:
                continue
            pick = int(ranked[0])
            if pick not in selected_set:
                selected_local.append(pick)
                selected_set.add(pick)
                per_cluster_counts[cid] += 1
            if len(selected_local) >= n_target:
                break

        # Second pass: fill respecting per-cluster cap
        ranked_all = np.argsort(score_pool)[::-1]
        for idx in ranked_all:
            idx = int(idx)
            if idx in selected_set:
                continue
            cid = int(cluster_of[idx])
            if per_cluster_counts[cid] >= self.max_per_cluster:
                continue
            selected_local.append(idx)
            selected_set.add(idx)
            per_cluster_counts[cid] += 1
            if len(selected_local) >= n_target:
                break

        # Third pass: fill any remaining regardless of cluster
        if len(selected_local) < n_target:
            for idx in ranked_all:
                idx = int(idx)
                if idx in selected_set:
                    continue
                selected_local.append(idx)
                selected_set.add(idx)
                if len(selected_local) >= n_target:
                    break

        return selected_local


# ===================================================================
# Variant 5: SubspaceStability
# ===================================================================

class SubspaceStability:
    """Correlated-subspace grouping on top of subsample stability selection.

    Runs :class:`SubsampleStability`, then analyses correlated feature groups
    among stable features and emits alternative equivalent feature sets by
    swapping within each correlated subspace.

    This is **not** a bootstrap variant — it is a post-processor that uses
    ``SubsampleStability`` as its inner engine.
    """

    def __init__(
        self,
        *,
        subsample_stability: SubsampleStability,
        corr_threshold: float = 0.85,
        selection_threshold: float = 0.6,
    ):
        self.subsample_stability = subsample_stability
        self.corr_threshold = corr_threshold
        self.selection_threshold = selection_threshold

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_target_features: int,
        prefilter_fn: Callable,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[int, float]]:
        """Run subsample stability + subspace post-processing."""
        base_results, base_scores = self.subsample_stability.run(
            X, y, n_target_features, prefilter_fn, **kwargs,
        )
        if not base_results or "selected_indices" not in base_results:
            return base_results, base_scores

        n_samples, n_features = X.shape
        if n_features == 0:
            return base_results, base_scores

        selection_freq = np.asarray(
            base_results.get("selection_frequency", np.zeros(n_features)), dtype=float,
        ).ravel()
        stability_score = np.asarray(
            base_results.get("stability_score", selection_freq), dtype=float,
        ).ravel()
        if selection_freq.size != n_features:
            padded = np.zeros(n_features, dtype=float)
            upto = int(min(n_features, selection_freq.size))
            padded[:upto] = selection_freq[:upto]
            selection_freq = padded
        if stability_score.size != n_features:
            padded = np.zeros(n_features, dtype=float)
            upto = int(min(n_features, stability_score.size))
            padded[:upto] = stability_score[:upto]
            stability_score = padded

        primary_selected = np.array(
            sorted(
                set(
                    int(i)
                    for i in np.asarray(
                        base_results.get("selected_indices", []), dtype=int,
                    ).tolist()
                )
            ),
            dtype=int,
        )
        if primary_selected.size == 0:
            return base_results, base_scores

        # --- candidate expansion ---
        stable_floor = float(max(0.10, 0.75 * self.selection_threshold))
        candidate_idx = np.where(selection_freq >= stable_floor)[0]
        if candidate_idx.size < 2:
            top_cap = int(min(n_features, max(8, 3 * int(max(1, n_target_features)))))
            candidate_idx = np.argsort(stability_score)[::-1][:top_cap]
        candidate_idx = np.unique(
            np.concatenate(
                [
                    np.asarray(candidate_idx, dtype=int),
                    np.asarray(primary_selected, dtype=int),
                ]
            )
        )
        if candidate_idx.size < 2:
            enriched = dict(base_results)
            enriched["equivalent_models"] = [np.asarray(primary_selected, dtype=int)]
            enriched["subspace_groups"] = []
            enriched["subspace_group_sizes"] = []
            enriched["n_subspace_groups"] = 0
            enriched["subspace_corr_threshold"] = float(self.corr_threshold)
            return enriched, base_scores

        # --- correlation clustering among candidates ---
        X_candidate = X[:, candidate_idx]
        corr = np.corrcoef(X_candidate, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)
        abs_corr = np.abs(corr)
        corr_threshold = float(np.clip(self.corr_threshold, 0.35, 0.99))
        adj = abs_corr >= corr_threshold
        np.fill_diagonal(adj, True)

        visited = np.zeros(candidate_idx.size, dtype=bool)
        components: List[np.ndarray] = []
        for start in range(candidate_idx.size):
            if visited[start]:
                continue
            stack = [int(start)]
            visited[start] = True
            comp: List[int] = []
            while stack:
                node = int(stack.pop())
                comp.append(node)
                neighbors = np.where(adj[node])[0]
                for nb in neighbors:
                    nb = int(nb)
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(np.asarray(comp, dtype=int))

        # --- build correlated feature groups ---
        candidate_score_lookup = {
            int(idx): float(stability_score[int(idx)]) for idx in candidate_idx.tolist()
        }
        group_features: List[np.ndarray] = []
        feature_to_group: Dict[int, int] = {}
        for _gid, comp in enumerate(components):
            local = np.asarray(comp, dtype=int)
            if local.size < 2:
                continue
            feats = candidate_idx[local]
            feats = feats[
                np.argsort([candidate_score_lookup[int(f)] for f in feats])[::-1]
            ]
            if feats.size < 2:
                continue
            group_features.append(np.asarray(feats, dtype=int))
            current_gid = len(group_features) - 1
            for feat in feats:
                feature_to_group[int(feat)] = current_gid

        # --- generate equivalent models ---
        equivalent_models: List[np.ndarray] = [np.asarray(primary_selected, dtype=int)]
        max_models = 3
        ranked_all = np.argsort(stability_score)[::-1]

        for alt_rank in range(1, max_models):
            variant = np.asarray(primary_selected, dtype=int).copy()
            changed = False
            for i, feat in enumerate(variant.tolist()):
                gid = feature_to_group.get(int(feat))
                if gid is None:
                    continue
                group = group_features[gid]
                if group.size <= alt_rank:
                    continue
                replacement = int(group[alt_rank])
                if replacement == int(feat) or replacement in variant:
                    continue
                variant[i] = replacement
                changed = True
            if not changed:
                continue

            dedup: List[int] = []
            for feat in variant.tolist():
                if feat not in dedup:
                    dedup.append(int(feat))
            for feat in ranked_all.tolist():
                if len(dedup) >= int(n_target_features):
                    break
                feat_int = int(feat)
                if feat_int not in dedup:
                    dedup.append(feat_int)
            dedup_arr = np.asarray(dedup[: int(n_target_features)], dtype=int)
            if dedup_arr.size == 0:
                continue
            duplicate = any(
                np.array_equal(dedup_arr, existing) for existing in equivalent_models
            )
            if not duplicate:
                equivalent_models.append(dedup_arr)

        # --- enrich results ---
        enriched_results = dict(base_results)
        enriched_results["equivalent_models"] = [
            np.asarray(model, dtype=int) for model in equivalent_models
        ]
        enriched_results["subspace_groups"] = [
            np.asarray(group, dtype=int) for group in group_features
        ]
        enriched_results["subspace_group_sizes"] = [
            int(group.size) for group in group_features
        ]
        enriched_results["n_subspace_groups"] = int(len(group_features))
        enriched_results["subspace_corr_threshold"] = float(corr_threshold)
        return enriched_results, base_scores

"""
Copula-based knock-off feature selector
======================================

Implements Algorithm 1 (TDCK) and Algorithm 2 (DTDCKe) from
Roman-Vasquez et al. (2024) using the *pyvinecopulib* package (>=0.7.5).

The paper uses vine copulas to model multivariate dependence and generate
knockoffs via Rosenblatt transforms + Gaussian bridge perturbation.
This ensures (approximate) exchangeability, which is required for FDR
control in the knockoff filter framework.

Works for both classification and regression – the response ``y`` is
treated as numeric; for classification pass the class labels (0/1, …).
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Optional

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import check_random_state
from tqdm.auto import tqdm
from scipy import stats as sps

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  pyvinecopulib (C++ backend by Nagler & Vatter, >=0.7.5 required)  #
# ------------------------------------------------------------------ #
try:
    from pyvinecopulib import (
        Vinecop,
        DVineStructure,
        FitControlsVinecop,
    )
except ImportError:
    Vinecop = None
    DVineStructure = None
    FitControlsVinecop = None


# -------------------------- utilities ----------------------------- #


class _Deadline:
    """Soft wall-clock deadline used to bound vine fit + draw aggregation."""

    def __init__(self, budget_seconds: Optional[float]) -> None:
        if budget_seconds is None or float(budget_seconds) <= 0.0:
            self._end: Optional[float] = None
        else:
            self._end = time.monotonic() + float(budget_seconds)

    @property
    def enabled(self) -> bool:
        return self._end is not None

    def remaining(self) -> Optional[float]:
        if self._end is None:
            return None
        return float(self._end - time.monotonic())

    def expired(self) -> bool:
        if self._end is None:
            return False
        return time.monotonic() >= self._end


def _safe_truncation_level(provided: Optional[int], p: int) -> int:
    """Resolve a safe D-vine truncation level for ``p`` features.

    Caps at ``ceil(log2(p)) + 2`` (≥ 5 for any p ≥ 8) to defend against the
    historical ``trunc_lvl = p - 1`` blow-up that fits a full untruncated vine
    on HDLSS data (O(p²) pair copulas with BIC family selection).
    """
    p = int(max(1, p))
    if p <= 1:
        return 1
    log_cap = int(max(5, math.ceil(math.log2(max(2, p))) + 2))
    if provided is None:
        return int(min(log_cap, p - 1))
    return int(min(max(1, int(provided)), log_cap, p - 1))


def _greedy_tsp_ordering(U: np.ndarray) -> list[int]:
    """Greedy nearest-neighbour TSP ordering on |Spearman ρ|.

    Vectorised: avoids the O(p²) interpreter-level inner loop. We compute
    ``|corrcoef(U)|`` once, then walk the path by ``argmax`` over a masked
    row, masking visited columns with ``-inf`` between picks.

    Returns 1-based indices suitable for ``DVineStructure(order=...)``.
    """
    p = U.shape[1]
    if p <= 1:
        return [1]
    rho = np.abs(np.corrcoef(U, rowvar=False))
    np.fill_diagonal(rho, 0.0)
    visited = np.empty(p, dtype=np.int64)
    visited[0] = 0
    remaining = np.ones(p, dtype=bool)
    remaining[0] = False
    for k in range(1, p):
        scores = np.where(remaining, rho[visited[k - 1]], -np.inf)
        nxt = int(np.argmax(scores))
        visited[k] = nxt
        remaining[nxt] = False
    return (visited + 1).tolist()


def _to_uniform(x: np.ndarray) -> np.ndarray:
    """
    Probability integral transform  →  i.i.d. U(0,1) margins.
    """
    n_quantiles = int(min(1000, max(10, x.shape[0] // 5)))
    qt = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="uniform",
        subsample=20_000,
        random_state=0,
        copy=True,
    )
    return qt.fit_transform(x)


def _lcd_stat(X, Xtilde, y, cv_kwargs):
    """
    Lasso-coefficient-difference (LCD) statistic from Candès et al. (2018).

    Sane defaults for ``max_iter`` / ``tol`` are applied unless the caller
    overrides them, so HDLSS data does not silently exhaust the default
    sklearn iteration cap inside the M-draw loop.
    """
    cv_kwargs = dict(cv_kwargs or {})
    cv_kwargs.setdefault("max_iter", 5000)
    cv_kwargs.setdefault("tol", 1e-3)
    X_aug = np.hstack([X, Xtilde])
    lasso = LassoCV(cv=5, **cv_kwargs).fit(X_aug, y)
    coefs = lasso.coef_
    p = X.shape[1]
    return np.abs(coefs[:p]) - np.abs(coefs[p:])


def _knockoff_threshold(W, alpha):
    """
    Adaptive knock-off threshold (eq. 2 in Barber & Candès 2015).
    """
    t_grid = np.sort(np.abs(W))[::-1]
    t_grid = t_grid[t_grid > 0]  # Exclude t=0 per Barber & Candès (2015)
    if t_grid.size == 0:
        return float("inf")
    ratio = (1 + (W[:, None] <= -t_grid).sum(0)) / np.maximum(
        1, (W[:, None] >= t_grid).sum(0)
    )
    ok = np.where(ratio <= alpha)[0]
    return float("inf") if ok.size == 0 else float(t_grid[ok[-1]])


def _ebh_support(e_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    e-BH support selection from averaged e-values.
    """
    e_vals = np.asarray(e_values, dtype=float).ravel()
    if e_vals.size == 0:
        return np.array([], dtype=int)

    e_vals = np.nan_to_num(e_vals, nan=0.0, posinf=1e12, neginf=0.0)
    p = int(e_vals.size)
    order = np.argsort(e_vals)[::-1]
    e_sorted = e_vals[order]
    thresh = (np.arange(1, p + 1, dtype=float) * float(alpha)) / float(p)
    thresh = np.maximum(thresh, 1e-12)
    valid = np.where(e_sorted >= (1.0 / thresh))[0]
    if valid.size == 0:
        return np.array([], dtype=int)
    k_hat = int(valid.max())
    return np.sort(order[: k_hat + 1])


# ------------------------ main selector --------------------------- #

class CopulaKnockoffSelector:
    """
    Derandomised TDCKe feature selector.

    Parameters
    ----------
    M : int, default=30
        Number of knock-off draws in the e-value aggregation loop.
    alpha_kn : float, default=0.1
        Target FDR level for each knock-off run.
    alpha_ebh : float, default=0.2
        Target FDR level for the e-BH procedure across M runs.
    truncation_level : int or None, default=None
        Optional D-vine truncation level for the p-dimensional vine of X.
        ``None`` means "use the safe log-scaled cap" (≥ 5, ≤ p-1) — see
        :func:`_safe_truncation_level`. The historical behaviour of
        ``None → p - 1`` (full untruncated vine) caused multi-day hangs on
        HDLSS data and is no longer permitted.
    conditional_bridge_rho : float, default=0.50
        Correlation used in Gaussian bridge sampling in Rosenblatt space.
        Lower values produce more perturbed knockoffs; higher values keep
        knockoffs closer to originals.
    generator : str, default="copula"
        Knockoff generator backend. ``"copula"`` uses DTDCKe with vine copulas.
        ``"deepdrk"`` uses a CPU-only low-rank residual sampler inspired by
        DeepDRK-style generative knockoffs.
    deepdrk_latent_fraction : float, default=0.35
        Fraction of latent SVD components used by the deepdrk generator.
    deepdrk_noise_scale : float, default=1.0
        Residual noise scale used by the deepdrk generator.
    show_progress : bool, default=False
        If True, renders a tqdm progress bar for the M draws.
    vine_kwargs : dict, optional
        Extra arguments forwarded to ``FitControlsVinecop`` (e.g. ``family_set``).
        ``num_threads`` is set from ``vine_num_threads`` and may be overridden
        here if needed.
    random_state : int or None
    time_budget_seconds : float, optional
        Soft wall-clock deadline (seconds). Once exceeded, the M-draw loop
        aborts at the next iteration boundary and ``fit`` returns with the
        e-values aggregated over the completed draws (or empty if the budget
        expired during the initial vine fit). ``None``/``<=0`` disables the
        budget. Note: SIGALRM cannot reliably interrupt the C++ ``Vinecop``
        kernel, so the budget is enforced *between* steps, not within them —
        the ``truncation_level`` cap above is the primary safeguard against
        runaway vine fits.
    vine_num_threads : int, optional
        Number of threads for the C++ vine fit. Defaults to
        ``min(8, cpu_count())``.
    lasso_max_iter : int, default=5000
        Coordinate-descent iteration cap for the per-draw ``LassoCV``.
    lasso_n_jobs : int, default=1
        ``n_jobs`` forwarded to ``LassoCV``. Set to ``-1`` to parallelise the
        per-alpha path; leave at 1 when running multiple selectors in parallel
        already.
    log_progress_every : int, default=5
        Emit ``logger.info`` progress every N completed draws when
        ``show_progress=False``. Set ``<= 0`` to disable.
    """

    def __init__(
        self,
        M: int = 30,
        alpha_kn: float = 0.1,
        alpha_ebh: float = 0.2,
        truncation_level: Optional[int] = None,
        conditional_bridge_rho: float = 0.50,
        generator: str = "copula",
        deepdrk_latent_fraction: float = 0.35,
        deepdrk_noise_scale: float = 1.0,
        show_progress: bool = False,
        vine_kwargs: Optional[dict] = None,
        random_state: Optional[int] = None,
        time_budget_seconds: Optional[float] = None,
        vine_num_threads: Optional[int] = None,
        lasso_max_iter: int = 5000,
        lasso_n_jobs: int = 1,
        log_progress_every: int = 5,
    ):
        self.M = int(max(1, M))
        self.alpha_kn = float(np.clip(alpha_kn, 1e-6, 0.99))
        self.alpha_ebh = float(np.clip(alpha_ebh, 1e-6, 0.99))
        self.truncation_level = (
            None if truncation_level is None else int(max(1, truncation_level))
        )
        self.conditional_bridge_rho = float(np.clip(conditional_bridge_rho, 0.05, 0.95))
        self.generator = str(generator or "copula").strip().lower()
        if self.generator not in {"copula", "deepdrk"}:
            self.generator = "copula"
        self.deepdrk_latent_fraction = float(np.clip(deepdrk_latent_fraction, 0.05, 1.0))
        self.deepdrk_noise_scale = float(max(0.0, deepdrk_noise_scale))
        self.show_progress = bool(show_progress)
        self.vine_kwargs = dict(vine_kwargs or {})
        self.random_state = random_state
        if time_budget_seconds is None or float(time_budget_seconds) <= 0.0:
            self.time_budget_seconds: Optional[float] = None
        else:
            self.time_budget_seconds = float(time_budget_seconds)
        if vine_num_threads is None:
            self.vine_num_threads = int(max(1, min(8, (os.cpu_count() or 1))))
        else:
            self.vine_num_threads = int(max(1, vine_num_threads))
        self.lasso_max_iter = int(max(100, lasso_max_iter))
        self.lasso_n_jobs = int(lasso_n_jobs) if lasso_n_jobs is not None else 1
        self.log_progress_every = int(log_progress_every)
        self.support_ = np.array([], dtype=int)
        self.e_avg_ = np.array([], dtype=float)
        self.low_information_diagnostics_ = {
            "n_samples": 0,
            "n_features": 0,
            "n_effective": 0,
            "n_nonzero_e_values": 0,
            "n_support": 0,
            "reason_code": "uninitialized",
            "completed_draws": 0,
            "requested_draws": int(self.M),
            "time_budget_exhausted": False,
            "inverse_rosenblatt_failures": 0,
        }
        self.truncation_level_effective_ = {
            "vine_x": None,
            "vine_2p": None,
            "conditional_bridge_rho": self.conditional_bridge_rho,
            "generator": self.generator,
            "deepdrk_latent_fraction": self.deepdrk_latent_fraction,
            "deepdrk_noise_scale": self.deepdrk_noise_scale,
        }

    def _resolve_truncation_level(self, p: int) -> int:
        return _safe_truncation_level(self.truncation_level, p)

    @staticmethod
    def _gaussian_bridge_uniform(v: np.ndarray, rng: np.random.RandomState, rho: float) -> np.ndarray:
        """
        Correlated uniform bridge in Rosenblatt space (single-row API).

        Kept for backward compatibility. Prefer
        :meth:`_gaussian_bridge_uniform_matrix` for the M-draw inner loop.
        """
        eps_u = 1e-6
        v_clip = np.clip(np.asarray(v, dtype=float).ravel(), eps_u, 1.0 - eps_u)
        z = sps.norm.ppf(v_clip)
        z = np.nan_to_num(z, nan=0.0, posinf=6.0, neginf=-6.0)
        noise = rng.normal(loc=0.0, scale=1.0, size=z.shape[0])
        bridge = float(np.sqrt(max(1e-8, 1.0 - float(rho) ** 2)))
        z_tilde = float(rho) * z + bridge * noise
        v_tilde = sps.norm.cdf(z_tilde)
        return np.clip(np.asarray(v_tilde, dtype=float).ravel(), eps_u, 1.0 - eps_u)

    @staticmethod
    def _gaussian_bridge_uniform_matrix(
        V: np.ndarray, rng: np.random.RandomState, rho: float
    ) -> np.ndarray:
        """Vectorised Gaussian bridge over a full ``(n, p)`` Rosenblatt block."""
        eps_u = 1e-6
        v_clip = np.clip(np.asarray(V, dtype=float), eps_u, 1.0 - eps_u)
        z = sps.norm.ppf(v_clip)
        z = np.nan_to_num(z, nan=0.0, posinf=6.0, neginf=-6.0)
        noise = rng.normal(loc=0.0, scale=1.0, size=z.shape)
        bridge = float(np.sqrt(max(1e-8, 1.0 - float(rho) ** 2)))
        z_tilde = float(rho) * z + bridge * noise
        v_tilde = sps.norm.cdf(z_tilde)
        return np.clip(np.asarray(v_tilde, dtype=float), eps_u, 1.0 - eps_u)

    @staticmethod
    def _deepdrk_generate_knockoffs(
        X: np.ndarray,
        rng: np.random.RandomState,
        *,
        latent_fraction: float,
        noise_scale: float,
        bridge_rho: float,
    ) -> np.ndarray:
        """CPU-only low-rank residual knockoff generator (DeepDRK-inspired)."""
        x_arr = np.asarray(X, dtype=float)
        n, p = x_arr.shape
        if n <= 1 or p <= 0:
            return np.asarray(x_arr, dtype=float)

        mean = np.mean(x_arr, axis=0, keepdims=True)
        centered = x_arr - mean
        try:
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
        except Exception:
            noise = rng.normal(loc=0.0, scale=1e-3, size=x_arr.shape)
            return np.asarray(x_arr + noise, dtype=float)

        rank_max = int(max(1, min(centered.shape[0], centered.shape[1])))
        latent_k = int(max(1, min(rank_max, round(float(latent_fraction) * rank_max))))
        low_rank = (u[:, :latent_k] * s[:latent_k]) @ vt[:latent_k, :]
        residual = centered - low_rank
        perm = rng.permutation(n)
        residual_perm = residual[perm]
        residual_std = np.std(residual, axis=0, ddof=1) if n > 1 else np.std(residual, axis=0)
        residual_std = np.nan_to_num(residual_std, nan=0.0, posinf=0.0, neginf=0.0)
        gaussian_noise = rng.normal(loc=0.0, scale=residual_std, size=centered.shape)
        bridge = float(np.clip(1.0 - bridge_rho, 0.05, 0.95))
        knock_centered = low_rank + bridge * residual_perm + float(noise_scale) * (1.0 - bridge) * gaussian_noise
        return np.asarray(knock_centered + mean, dtype=float)

    @staticmethod
    def _build_vine_controls(
        trunc_lvl: int, num_threads: int, vine_kwargs: dict
    ) -> "FitControlsVinecop":
        """Build ``FitControlsVinecop`` with safe defaults.

        ``num_threads`` is included only if not already supplied via
        ``vine_kwargs`` and only if the local pyvinecopulib build accepts it
        (older builds will raise ``TypeError`` — we then drop the kwarg).
        """
        controls_kw = dict(
            trunc_lvl=int(trunc_lvl),
            selection_criterion="bic",
        )
        if "num_threads" not in vine_kwargs and num_threads and num_threads > 1:
            controls_kw["num_threads"] = int(num_threads)
        controls_kw.update(vine_kwargs)
        try:
            return FitControlsVinecop(**controls_kw)
        except TypeError:
            controls_kw.pop("num_threads", None)
            return FitControlsVinecop(**controls_kw)

    # ------------------------ public API ---------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Compute e-values and selected support.

        *You may pass any numerical X – the method will rank-transform it
        to U(0,1) internally, so previous standardisation does **not**
        hurt.*

        Aborts at the next M-loop iteration boundary if
        ``time_budget_seconds`` elapses; aggregates the e-values over the
        completed draws and reports ``reason_code='time_budget_exhausted'``
        in :pyattr:`low_information_diagnostics_`.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        n, p = X_arr.shape
        e_mat = np.zeros((self.M, p))
        deadline = _Deadline(self.time_budget_seconds)
        time_budget_exhausted = False
        completed_draws = 0
        inverse_failures = 0

        use_copula = self.generator == "copula"
        if use_copula and Vinecop is None:
            raise ImportError(
                "CopulaKnockoffSelector requires `pyvinecopulib>=0.7.5` when "
                "generator='copula'. Install it with: pip install 'pyvinecopulib>=0.7.5'. "
                "Use generator='deepdrk' for CPU-only fallback."
            )

        if use_copula:
            # 1) put the features on the copula scale
            X_u = _to_uniform(X_arr)
            trunc_x = self._resolve_truncation_level(p)
            self.truncation_level_effective_ = {
                "vine_x": int(trunc_x),
                "vine_2p": None,
                "conditional_bridge_rho": float(self.conditional_bridge_rho),
                "generator": "copula",
                "deepdrk_latent_fraction": self.deepdrk_latent_fraction,
                "deepdrk_noise_scale": self.deepdrk_noise_scale,
                "vine_num_threads": int(self.vine_num_threads),
                "truncation_level_requested": (
                    None if self.truncation_level is None else int(self.truncation_level)
                ),
            }

            # Greedy TSP ordering (Algorithm 1, step 1 of Roman-Vasquez 2024).
            t0_order = time.monotonic()
            order = _greedy_tsp_ordering(X_u)
            t_order = time.monotonic() - t0_order

            controls = self._build_vine_controls(
                trunc_lvl=int(trunc_x),
                num_threads=int(self.vine_num_threads),
                vine_kwargs=self.vine_kwargs,
            )
            dvine = DVineStructure(order=order)

            logger.info(
                "Copula knockoff: fitting D-vine (n=%d, p=%d, trunc_lvl=%d, threads=%d, ordering=%.1fs)",
                n, p, int(trunc_x), int(self.vine_num_threads), t_order,
            )
            t0_vine = time.monotonic()
            vine_X = Vinecop.from_data(X_u, structure=dvine, controls=controls)
            t_vine = time.monotonic() - t0_vine
            logger.info("Copula knockoff: D-vine fit complete in %.1fs", t_vine)

            if deadline.expired():
                logger.warning(
                    "Copula knockoff: time budget exhausted after vine fit (%.1fs); "
                    "returning empty support.",
                    t_vine,
                )
                self.e_avg_ = np.zeros(p, dtype=float)
                self.support_ = np.array([], dtype=int)
                self._record_diagnostics(
                    n=n, p=p, completed_draws=0,
                    reason_code="time_budget_exhausted",
                    time_budget_exhausted=True,
                    inverse_failures=0,
                )
                return self
        else:
            X_u = None
            order = None
            vine_X = None
            self.truncation_level_effective_ = {
                "vine_x": None,
                "vine_2p": None,
                "conditional_bridge_rho": float(self.conditional_bridge_rho),
                "generator": "deepdrk",
                "deepdrk_latent_fraction": self.deepdrk_latent_fraction,
                "deepdrk_noise_scale": self.deepdrk_noise_scale,
            }

        if isinstance(self.random_state, np.random.RandomState):
            fit_rng = np.random.RandomState()
            fit_rng.set_state(self.random_state.get_state())
        else:
            fit_rng = check_random_state(self.random_state)

        max_seed = int(2**31 - 1)
        # Avoid sampling-without-replacement over a gigantic integer range.
        # `choice(max_seed, replace=False)` is prohibitively slow and can
        # trigger large transient allocations.
        rng_seeds = fit_rng.randint(0, max_seed, size=self.M)

        for m, seed in enumerate(
            tqdm(rng_seeds, desc="DTDCKe", disable=not self.show_progress, leave=False)
        ):
            if deadline.expired():
                logger.warning(
                    "Copula knockoff: time budget exhausted after %d/%d draws; "
                    "aggregating partial results.",
                    completed_draws, self.M,
                )
                time_budget_exhausted = True
                break

            seed_int = int(seed)
            rng = np.random.RandomState(seed_int)

            if use_copula:
                # Batched Rosenblatt: pyvinecopulib accepts the full (n, p) matrix.
                # Falls back to a row-by-row loop only if the batch call raises,
                # so a single bad row cannot poison the entire draw.
                V = vine_X.rosenblatt(X_u)
                V_tilde = self._gaussian_bridge_uniform_matrix(
                    V, rng, rho=self.conditional_bridge_rho
                )
                try:
                    U_tilde_raw = vine_X.inverse_rosenblatt(V_tilde)
                    row_failures = 0
                except Exception as exc:  # noqa: BLE001 — pyvinecopulib raises generic errors
                    logger.warning(
                        "Copula knockoff: batched inverse_rosenblatt failed on draw %d "
                        "(%s); falling back to per-row mode.",
                        m, type(exc).__name__,
                    )
                    U_tilde_raw, row_failures = self._inverse_rosenblatt_per_row(
                        vine_X, V_tilde
                    )

                if row_failures > 0:
                    inverse_failures += int(row_failures)
                    failure_frac = float(row_failures) / float(max(1, n))
                    if failure_frac > 0.20:
                        logger.warning(
                            "Copula knockoff: skipping draw %d (%d/%d rows failed "
                            "inverse_rosenblatt; %.0f%% > 20%% threshold).",
                            m, row_failures, n, failure_frac * 100.0,
                        )
                        continue

                U_tilde = np.clip(np.asarray(U_tilde_raw, dtype=float), 1e-6, 1.0 - 1e-6)

                # empirical inverse CDF (type 8)
                X_tilde = np.stack(
                    [
                        np.quantile(X_arr[:, j], U_tilde[:, j], method="median_unbiased")
                        for j in range(p)
                    ],
                    axis=1,
                )
            else:
                X_tilde = self._deepdrk_generate_knockoffs(
                    X_arr,
                    rng,
                    latent_fraction=self.deepdrk_latent_fraction,
                    noise_scale=self.deepdrk_noise_scale,
                    bridge_rho=self.conditional_bridge_rho,
                )

            # --------------------------------------------------
            #   W-statistics & e-values
            # --------------------------------------------------
            cv_kwargs = dict(
                random_state=seed_int,
                max_iter=int(self.lasso_max_iter),
                n_jobs=int(self.lasso_n_jobs),
            )
            W = _lcd_stat(X_arr, X_tilde, y_arr, cv_kwargs=cv_kwargs)
            T = _knockoff_threshold(W, alpha=self.alpha_kn)
            e_mat[m] = p * (W >= T) / (1 + (W <= -T).sum())
            completed_draws += 1

            if (
                self.log_progress_every > 0
                and (m + 1) % self.log_progress_every == 0
                and m + 1 < self.M
            ):
                rem = deadline.remaining()
                rem_str = f", budget_remaining={rem:.0f}s" if rem is not None else ""
                logger.info(
                    "Copula knockoff: draw %d/%d complete (failed=%d%s)",
                    m + 1, self.M, inverse_failures, rem_str,
                )

        # 2)  aggregate e-values and run e-BH
        if completed_draws > 0:
            self.e_avg_ = e_mat[:completed_draws].mean(0) if time_budget_exhausted else e_mat.mean(0)
        else:
            self.e_avg_ = np.zeros(p, dtype=float)
        self.support_ = _ebh_support(self.e_avg_, alpha=self.alpha_ebh)
        n_nonzero = int(np.sum(np.asarray(self.e_avg_, dtype=float) > 1e-12))
        if time_budget_exhausted and completed_draws == 0:
            reason_code = "time_budget_exhausted"
        elif self.support_.size == 0:
            if n_nonzero == 0:
                reason_code = "all_zero_e_values"
            else:
                reason_code = "ebh_empty_support"
        elif time_budget_exhausted:
            reason_code = "ok_partial_time_budget"
        else:
            reason_code = "ok"
        self._record_diagnostics(
            n=n, p=p, completed_draws=completed_draws,
            reason_code=reason_code,
            time_budget_exhausted=time_budget_exhausted,
            inverse_failures=inverse_failures,
        )
        return self

    @staticmethod
    def _inverse_rosenblatt_per_row(
        vine_X, V_tilde: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Row-by-row fallback for ``inverse_rosenblatt``.

        Records the number of rows that fall back to the copula-scale
        ``v_tilde`` (instead of the natural-scale ``u_tilde``) when the C++
        inversion raises. Caller is expected to skip the draw entirely when
        the failure fraction is too large.
        """
        n = V_tilde.shape[0]
        out = np.empty_like(V_tilde, dtype=float)
        failures = 0
        for i in range(n):
            try:
                out[i] = np.asarray(
                    vine_X.inverse_rosenblatt(V_tilde[i:i + 1]), dtype=float
                ).ravel()
            except Exception:  # noqa: BLE001
                out[i] = V_tilde[i]
                failures += 1
        return out, failures

    def _record_diagnostics(
        self, *, n: int, p: int, completed_draws: int, reason_code: str,
        time_budget_exhausted: bool, inverse_failures: int,
    ) -> None:
        self.low_information_diagnostics_ = {
            "n_samples": int(n),
            "n_features": int(p),
            "n_effective": int(min(n, p)),
            "n_nonzero_e_values": int(np.sum(np.asarray(self.e_avg_, dtype=float) > 1e-12)),
            "n_support": int(self.support_.size),
            "reason_code": str(reason_code),
            "completed_draws": int(completed_draws),
            "requested_draws": int(self.M),
            "time_budget_exhausted": bool(time_budget_exhausted),
            "inverse_rosenblatt_failures": int(inverse_failures),
        }

    # ---------------- helper accessors ----------------------- #

    def get_support(self) -> np.ndarray:
        """Return indices of selected features (may be empty)."""
        return self.support_

    def get_weights(self, eps: float = 1e-6) -> np.ndarray:
        """
        Normalise averaged e-values to [0, 1] – handy as *soft* weights.
        """
        w = self.e_avg_.copy()
        w /= w.max() + eps
        return w

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

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import check_random_state
from tqdm.auto import tqdm
from scipy import stats as sps

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


def _greedy_tsp_ordering(U: np.ndarray) -> list[int]:
    """Greedy nearest-neighbour TSP ordering on |Spearman ρ|.

    Produces a Hamiltonian path through the variables that places
    strongly correlated variables adjacent — exactly the ordering that
    Roman-Vasquez et al. (2024) use for the D-vine (their Algorithm 1,
    step 1).  Spearman is used instead of Kendall because it is O(n log n)
    per pair rather than O(n²), which matters when p is large.

    Returns 1-based indices suitable for ``DVineStructure(order=...)``.
    """
    p = U.shape[1]
    if p <= 1:
        return [1]
    rho = np.abs(np.corrcoef(U, rowvar=False))     # Pearson on the ranks ≈ Spearman
    np.fill_diagonal(rho, 0.0)
    visited = [0]
    remaining = set(range(1, p))
    while remaining:
        last = visited[-1]
        best = max(remaining, key=lambda j: rho[last, j])
        visited.append(best)
        remaining.remove(best)
    return [v + 1 for v in visited]                 # 1-based for pyvinecopulib


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
    """
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
        Extra arguments forwarded to ``Vinecop`` (e.g. ``family_set``).
    random_state : int or None
    """

    def __init__(
        self,
        M: int = 30,
        alpha_kn: float = 0.1,
        alpha_ebh: float = 0.2,
        truncation_level: int | None = None,
        conditional_bridge_rho: float = 0.50,
        generator: str = "copula",
        deepdrk_latent_fraction: float = 0.35,
        deepdrk_noise_scale: float = 1.0,
        show_progress: bool = False,
        vine_kwargs: dict | None = None,
        random_state: int | None = None,
    ):
        self.M = int(max(1, M))
        self.alpha_kn = float(np.clip(alpha_kn, 1e-6, 0.99))
        self.alpha_ebh = float(np.clip(alpha_ebh, 1e-6, 0.99))
        self.truncation_level = None if truncation_level is None else int(max(1, truncation_level))
        self.conditional_bridge_rho = float(np.clip(conditional_bridge_rho, 0.05, 0.95))
        self.generator = str(generator or "copula").strip().lower()
        if self.generator not in {"copula", "deepdrk"}:
            self.generator = "copula"
        self.deepdrk_latent_fraction = float(np.clip(deepdrk_latent_fraction, 0.05, 1.0))
        self.deepdrk_noise_scale = float(max(0.0, deepdrk_noise_scale))
        self.show_progress = bool(show_progress)
        self.vine_kwargs = vine_kwargs or {}
        self.random_state = random_state
        self.support_ = np.array([], dtype=int)
        self.e_avg_ = np.array([], dtype=float)
        self.low_information_diagnostics_ = {
            "n_samples": 0,
            "n_features": 0,
            "n_effective": 0,
            "n_nonzero_e_values": 0,
            "n_support": 0,
            "reason_code": "uninitialized",
        }
        self.truncation_level_effective_ = {
            "vine_x": None,
            "vine_2p": None,
            "conditional_bridge_rho": self.conditional_bridge_rho,
            "generator": self.generator,
            "deepdrk_latent_fraction": self.deepdrk_latent_fraction,
            "deepdrk_noise_scale": self.deepdrk_noise_scale,
        }

    def _resolve_truncation_level(self, p: int) -> int | None:
        p = int(max(1, p))
        if p <= 1:
            return None
        if self.truncation_level is None:
            return None
        return int(np.clip(self.truncation_level, 1, p - 1))

    @staticmethod
    def _gaussian_bridge_uniform(v: np.ndarray, rng: np.random.RandomState, rho: float) -> np.ndarray:
        """
        Correlated uniform bridge in Rosenblatt space.

        Given a row-wise Rosenblatt coordinate `v` and `rho in (0,1)`,
        samples `v_tilde` with Gaussian copula coupling:
            Z_tilde = rho * Z + sqrt(1-rho^2) * eps,  eps ~ N(0, I),
            v_tilde = Phi(Z_tilde).
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
        except Exception as exc:
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

    # ------------------------ public API ---------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Compute e-values and selected support.

        *You may pass any numerical X – the method will rank-transform it
        to U(0,1) internally, so previous standardisation does **not**
        hurt.*
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        n, p = X_arr.shape
        e_mat = np.zeros((self.M, p))

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
                "vine_x": trunc_x,
                "vine_2p": None,
                "conditional_bridge_rho": float(self.conditional_bridge_rho),
                "generator": "copula",
                "deepdrk_latent_fraction": self.deepdrk_latent_fraction,
                "deepdrk_noise_scale": self.deepdrk_noise_scale,
            }

            # Fit p-dimensional D-vine for X once per fit call.
            # Greedy TSP ordering (Algorithm 1, step 1 of Roman-Vasquez 2024).
            order = _greedy_tsp_ordering(X_u)
            dvine = DVineStructure(order=order)
            controls_kw = dict(
                trunc_lvl=trunc_x if trunc_x is not None else p - 1,
                selection_criterion="bic",
            )
            controls_kw.update(self.vine_kwargs)
            controls = FitControlsVinecop(**controls_kw)
            vine_X = Vinecop.from_data(X_u, structure=dvine, controls=controls)
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
            seed_int = int(seed)
            rng = np.random.RandomState(seed_int)

            if use_copula:
                # --------------------------------------------------
                #   sample knockoffs conditionally via fitted vine +
                #   Gaussian bridge in Rosenblatt space.
                #   pyvinecopulib handles column ordering internally
                #   via the DVineStructure, so data stays in natural
                #   column order.
                # --------------------------------------------------
                U_tilde = np.empty_like(X_u)
                for i in range(n):
                    u_row = X_u[i:i + 1]              # (1, p) natural order
                    v = vine_X.rosenblatt(u_row)[0]
                    v_tilde = self._gaussian_bridge_uniform(v, rng, rho=self.conditional_bridge_rho)
                    try:
                        u_tilde_row = vine_X.inverse_rosenblatt(v_tilde.reshape(1, -1))[0]
                    except Exception as exc:
                        # Conservative fallback: stay on copula scale if inversion fails.
                        u_tilde_row = v_tilde
                    U_tilde[i] = np.clip(
                        np.asarray(u_tilde_row, dtype=float).ravel(),
                        1e-6,
                        1.0 - 1e-6,
                    )

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
            W = _lcd_stat(X_arr, X_tilde, y_arr, cv_kwargs=dict(random_state=seed_int))
            T = _knockoff_threshold(W, alpha=self.alpha_kn)
            e_mat[m] = p * (W >= T) / (1 + (W <= -T).sum())

        # 2)  aggregate e-values and run e-BH
        self.e_avg_ = e_mat.mean(0)
        self.support_ = _ebh_support(self.e_avg_, alpha=self.alpha_ebh)
        n_nonzero = int(np.sum(np.asarray(self.e_avg_, dtype=float) > 1e-12))
        reason_code = "ok"
        if self.support_.size == 0:
            if n_nonzero == 0:
                reason_code = "all_zero_e_values"
            else:
                reason_code = "ebh_empty_support"
        self.low_information_diagnostics_ = {
            "n_samples": int(n),
            "n_features": int(p),
            "n_effective": int(min(n, p)),
            "n_nonzero_e_values": int(n_nonzero),
            "n_support": int(self.support_.size),
            "reason_code": str(reason_code),
        }
        return self

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

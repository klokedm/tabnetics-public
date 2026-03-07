import numpy as np
import scipy.stats as sps
import warnings
from typing import Dict, Tuple, List, Any

__all__ = [
    "BootstrapGOFSelector",
    "select_best_distribution_bootstrap",
    "select_best_distribution_bic",
    "auto_select_distribution",
]


class BootstrapGOFSelector:
    """Monte-Carlo goodness-of-fit selector tailored to *small* samples.

    This class recalibrates Kolmogorov--Smirnov or Cram\u00E9r-von Mises statistics
    via parametric bootstrapping.  For every candidate distribution we:
      1. Fit parameters (by MLE or SciPy default `fit`).
      2. Compute the observed test statistic.
      3. Simulate *n_boot* synthetic samples **of the same size** as the data,
         re-fitting the parameters for *each* replicate to account for
         estimation uncertainty.
      4. Estimate the p-value as the proportion of bootstrapped statistics that
         are **at least as extreme** as the observed statistic.

    An empirical p-value obtained this way is valid for tiny *n* because it
    respects parameter uncertainty and small-sample variability, unlike the
    asymptotic tables used by vanilla KS/CvM in SciPy.
    """

    SUPPORTED_STATS = {"ks", "cvm"}

    def __init__(self, statistic: str = "cvm", n_boot: int = 2000, random_state: Any = None):
        if statistic not in self.SUPPORTED_STATS:
            raise ValueError(f"statistic must be one of {self.SUPPORTED_STATS}")
        self.statistic = statistic
        self.n_boot = int(n_boot)
        self.rng = np.random.default_rng(random_state)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def p_value(self, data: np.ndarray, dist: sps.rv_continuous, params: Tuple) -> Tuple[float, float]:
        """Return (p-value, observed_statistic)."""
        obs_stat = self._compute_stat(data, dist, params)
        boot_stats = np.empty(self.n_boot)

        for i in range(self.n_boot):
            sim = dist.rvs(*params[:-2], loc=params[-2], scale=params[-1], size=data.size, random_state=self.rng)
            # Re-fit on each bootstrap sample to mimic real workflow
            try:
                sim_params = dist.fit(sim)
            except Exception as exc:
                # If re-fit fails, fall back to original params – conservative.
                sim_params = params
            boot_stats[i] = self._compute_stat(sim, dist, sim_params)

        p_val = (boot_stats >= obs_stat).mean()
        return p_val, obs_stat

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_stat(self, sample: np.ndarray, dist: sps.rv_continuous, params: Tuple) -> float:
        if self.statistic == "ks":
            return sps.kstest(sample, dist.cdf, args=params).statistic
        # default cvm
        return sps.cramervonmises(sample, dist.cdf, args=params).statistic


# -------------------------------------------------------------------------
#  Bootstrap-based selection
# -------------------------------------------------------------------------

def _default_dist_dict() -> Dict[str, sps.rv_continuous]:
    return {
        "norm": sps.norm,
        "expon": sps.expon,
        "weibull_min": sps.weibull_min,
        "gamma": sps.gamma,
        "lognorm": sps.lognorm,
        "beta": sps.beta,
        "uniform": sps.uniform,
    }


def select_best_distribution_bootstrap(
    data: np.ndarray,
    distributions: Dict[str, sps.rv_continuous] | None = None,
    statistic: str = "cvm",
    n_boot: int = 2000,
    random_state: Any = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (best_dist_name, sorted_result_list).

    Each result is a dict:  {"name", "p_value", "stat", "params"} sorted by
    descending p-value.  Best distribution is simply the first entry.
    """
    if distributions is None:
        distributions = _default_dist_dict()

    selector = BootstrapGOFSelector(statistic, n_boot, random_state)
    results: List[Dict[str, Any]] = []

    for name, dist in distributions.items():
        try:
            params = dist.fit(data)
            p_val, obs_stat = selector.p_value(data, dist, params)
            results.append({"name": name, "p_value": p_val, "stat": obs_stat, "params": params})
        except Exception as exc:
            warnings.warn(f"{name}: fit failed – {exc}")
            continue

    if not results:
        raise RuntimeError("No distribution could be fitted; cannot select best.")

    # Higher p-value ⇒ better fit
    results.sort(key=lambda d: d["p_value"], reverse=True)
    return results[0]["name"], results


# -------------------------------------------------------------------------
#  BIC — Bayesian Information Criterion (quick Bayesian proxy)
# -------------------------------------------------------------------------

def _bic(loglik: float, n: int, k: int) -> float:
    return -2.0 * loglik + k * np.log(n)


def select_best_distribution_bic(
    data: np.ndarray,
    distributions: Dict[str, sps.rv_continuous] | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (best_dist_name, sorted_result_list) using BIC."""
    if distributions is None:
        distributions = _default_dist_dict()

    n = data.size
    results: List[Dict[str, Any]] = []

    for name, dist in distributions.items():
        try:
            params = dist.fit(data)
            loglik = dist.logpdf(data, *params).sum()
            bic_val = _bic(loglik, n, len(params))
            results.append({"name": name, "bic": bic_val, "params": params})
        except Exception as exc:
            warnings.warn(f"{name}: fit failed – {exc}")
            continue

    if not results:
        raise RuntimeError("No distribution could be fitted; cannot select best.")

    # Lower BIC ⇒ better (more parsimonious) fit
    results.sort(key=lambda d: d["bic"])
    return results[0]["name"], results


# -------------------------------------------------------------------------
#  Auto strategy: small-n ⇒ bootstrap, else fall back
# -------------------------------------------------------------------------

def auto_select_distribution(
    data: np.ndarray,
    *,
    small_sample_threshold: int = 30,
    statistic: str = "cvm",
    n_boot: int = 2000,
    random_state: Any = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Dispatch to the most appropriate selection strategy.

    By default:
      • If n ≤ *small_sample_threshold* (30), use bootstrapped CvM.
      • Otherwise, pick the distribution with the lowest BIC.

    This heuristic blends a computationally intensive but accurate method for
    tiny datasets with a fast, information-criterion method for larger samples.
    """
    n = data.size
    if n <= small_sample_threshold:
        return select_best_distribution_bootstrap(data, statistic=statistic, n_boot=n_boot, random_state=random_state)
    return select_best_distribution_bic(data)


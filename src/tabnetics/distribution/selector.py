import numpy as np
import scipy.stats as sps
from scipy.stats import (kstest, cramervonmises, anderson, normaltest,
                         skew, kurtosis, shapiro, chi2)
from math import comb
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
from typing import Dict, Tuple, List, Optional, Any, Sequence
from dataclasses import dataclass
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
import os
import zlib
import logging  # Added for T-AUDIT-001-FIX-004

# FIX CRITICAL-002: Force spawn mode for ProcessPoolExecutor to avoid fork race hazards (T-AUDIT-001-FIX-002)
# This prevents shared memory contamination across RunPod shards when using parallel DF fitting.
# Must be called before any multiprocessing operations.
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # If start method is already set, this is acceptable
    pass

# Ignore common runtime warnings from stats calculations and UserWarnings from shapiro
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.stats._morestats')

# Set up logger for this module
logger = logging.getLogger(__name__)

_POSITIVE_ONLY_FAMILIES = {
    "expon",
    "gamma",
    "lognorm",
    "weibull_min",
    "pareto",
    "invweibull",
    "invgauss",
    "geninvgauss",
    "invgamma",
    "fisk",
    "genpareto",
    "gengamma",
}

try:
    from .mnpo_core import (
        aggregate_payoff_matrix as _mnpo_aggregate_payoff_matrix,
        apply_oracle_redundancy_penalty as _mnpo_apply_oracle_redundancy_penalty,
        fit_tritrust_weights as _mnpo_fit_tritrust_weights,
        matrix_from_scalar_scores as _mnpo_matrix_from_scalar_scores,
        mirror_descent_reference_regularized as _mnpo_mirror_descent_reference_regularized,
        normalize_vector_01 as _mnpo_normalize_vector_01,
        pairwise_pref_from_scalar as _mnpo_pairwise_pref_from_scalar,
        tremble_oracle_matrices as _mnpo_tremble_oracle_matrices,
    )
except Exception as exc:
    from tabnetics.core.mnpo import (  # type: ignore
        aggregate_payoff_matrix as _mnpo_aggregate_payoff_matrix,
        apply_oracle_redundancy_penalty as _mnpo_apply_oracle_redundancy_penalty,
        fit_tritrust_weights as _mnpo_fit_tritrust_weights,
        matrix_from_scalar_scores as _mnpo_matrix_from_scalar_scores,
        mirror_descent_reference_regularized as _mnpo_mirror_descent_reference_regularized,
        normalize_vector_01 as _mnpo_normalize_vector_01,
        pairwise_pref_from_scalar as _mnpo_pairwise_pref_from_scalar,
        tremble_oracle_matrices as _mnpo_tremble_oracle_matrices,
    )


@dataclass
class DistributionFeatures:
    """
    Represents statistical features extracted from data, used to guide distribution selection.
    These features help in applying heuristics and bonuses for more accurate fitting.
    """
    mean: float
    std: float
    skewness: float
    excess_kurtosis: float # Fisher's definition (normal dist has 0)
    cv: float  # Coefficient of variation (std/mean)
    
    # Quantile-based features
    q25: float    # 25th percentile
    median: float # 50th percentile (q50)
    q75: float    # 75th percentile
    iqr: float    # Interquartile range (q75 - q25)
    
    # Boolean indicators based on data properties
    is_positive: bool       # True if all data points are > 0
    frac_negative: float    # Fraction of values below zero in the original data
    is_symmetric: bool      # True if absolute skewness is low
    has_heavy_tails: bool   # True if excess kurtosis is high

    # Scores for specific distribution characteristics
    exponential_cv_score: float  # Measures how close CV is to 1.0 (characteristic of exponential)
    uniform_variance_score: float  # Compares actual variance to theoretical uniform variance
    lognormal_score: float         # p-value from normality test on log-transformed data

    # Additional robust statistics
    l_cv: float            # L-moment coefficient of variation
    l_skew: float          # L-skewness
    l_kurtosis: float      # L-kurtosis
    hazard_slope: float    # Approximate slope from Weibull plot (hazard trend)
    
    @classmethod
    def from_data(cls, data: np.ndarray):
        """
        Factory method to extract and calculate distribution features from an input data array.
        Handles NaN cleaning and cases with insufficient data.
        """
        data_clean = data[~np.isnan(data)] # Remove NaN values first
        
        # If insufficient data points, return neutral/default features
        if len(data_clean) < 10: 
            return cls(
                mean=0, std=0, skewness=0, excess_kurtosis=0, cv=float('inf'),
                q25=0, median=0, q75=0, iqr=0,
                is_positive=False, frac_negative=0.0, is_symmetric=True, has_heavy_tails=False,
                exponential_cv_score=0, uniform_variance_score=0, lognormal_score=0,
                l_cv=0.0, l_skew=0.0, l_kurtosis=0.0, hazard_slope=0.0
            )
            
        # Calculate basic statistics
        mean_val = np.mean(data_clean)
        std_val = np.std(data_clean, ddof=1) # ddof=1 for sample standard deviation
        skew_val = skew(data_clean)
        kurt_val = kurtosis(data_clean) # Fisher's (excess) kurtosis
        
        # Calculate quantiles
        q25, median_val, q75 = np.percentile(data_clean, [25, 50, 75])
        iqr = q75 - q25
        
        # Calculate Coefficient of Variation (CV)
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else float('inf')
        
        # Determine boolean indicators
        is_positive = np.all(data_clean > 1e-9) # Consider data strictly positive
        frac_negative = float(np.mean(data_clean < -1e-12))
        is_symmetric = abs(skew_val) < 0.5    # Threshold for symmetry
        has_heavy_tails = kurt_val > 2.0      # Threshold for heavy tails (excess kurtosis)

        # Calculate exponential CV score (closer to 1 is better)
        exponential_cv_score = 0.0
        if cv < float('inf'):
            exponential_cv_score = max(0.0, 1.0 - abs(cv - 1.0))
        
        # Calculate uniform variance score
        uniform_variance_score = 0.0
        if len(data_clean) > 1:
            data_range = np.max(data_clean) - np.min(data_clean)
            if data_range > 1e-10: # Avoid division by zero if all values are the same
                expected_uniform_var = data_range**2 / 12
                actual_var = np.var(data_clean, ddof=1)
                if expected_uniform_var > 1e-10: # Ensure expected variance is positive
                    var_ratio = actual_var / expected_uniform_var
                    uniform_variance_score = max(0.0, 1.0 - abs(var_ratio - 1.0))
                else: # If expected var is 0 (e.g., data_range is 0)
                    uniform_variance_score = 1.0 if actual_var < 1e-10 else 0.0
            elif np.var(data_clean, ddof=1) < 1e-10: # If range is 0 and var is 0
                 uniform_variance_score = 1.0
        
        # Calculate lognormal score (p-value from normality test on log-data)
        lognormal_score = 0.0
        if is_positive and len(data_clean) >=3: # Need at least 3 samples for log and tests
            try:
                log_data = np.log(data_clean)
                # Ensure log_data is finite and has variance for normality tests
                if np.all(np.isfinite(log_data)) and len(np.unique(log_data)) > 1:
                    if 3 <= len(log_data) <= 5000: # Shapiro-Wilk for smaller samples
                        _, p_value = shapiro(log_data)
                        lognormal_score = p_value
                    elif len(log_data) > 5000: # D'Agostino and Pearson's K^2 test for larger samples
                        # normaltest requires at least 8 samples by default in some scipy versions
                        if len(log_data) >= 8:
                            _, p_value = normaltest(log_data)
                            lognormal_score = p_value
            except Exception as exc: # Catch any error during log transformation or normality test
                lognormal_score = 0.0 # Default to 0 on error

        # L-moments for robust shape information
        def _l_moments(x):
            x_sorted = np.sort(x)
            n = len(x_sorted)
            if n < 4:
                return 0.0, 0.0, 0.0, 0.0
            def b_r(r):
                coef = [comb(k-1, r)/comb(n-1, r) for k in range(r+1, n+1)]
                return np.sum(np.array(coef) * x_sorted[r:]) / n
            b0 = b_r(0)
            b1 = b_r(1)
            b2 = b_r(2)
            b3 = b_r(3)
            l1 = b0
            l2 = 2*b1 - b0
            l3 = 6*b2 - 6*b1 + b0
            l4 = 20*b3 - 30*b2 + 12*b1 - b0
            return l1, l2, l3, l4

        l1, l2, l3, l4 = _l_moments(data_clean)
        l_cv_val = l2 / abs(l1) if abs(l1) > 1e-12 else 0.0
        l_skew_val = l3 / l2 if abs(l2) > 1e-12 else 0.0
        l_kurt_val = l4 / l2 if abs(l2) > 1e-12 else 0.0

        # Approximate hazard slope using Weibull plotting positions
        hazard_slope_val = 0.0
        try:
            positive_data = data_clean[data_clean > 0]
            if len(positive_data) >= 5:
                x_sorted = np.sort(positive_data)
                n_pos = len(x_sorted)
                ranks = np.arange(1, n_pos+1)
                F = ranks / (n_pos + 1.0)
                y = np.log(-np.log(1.0 - F))
                X = np.log(x_sorted)
                if np.all(np.isfinite(X)) and np.all(np.isfinite(y)):
                    slope, _ = np.polyfit(X, y, 1)
                    hazard_slope_val = float(slope)
        except Exception as exc:
            hazard_slope_val = 0.0

        return cls(
            mean=mean_val, std=std_val, skewness=skew_val, excess_kurtosis=kurt_val, cv=cv,
            q25=q25, median=median_val, q75=q75, iqr=iqr,
            is_positive=is_positive, frac_negative=frac_negative,
            is_symmetric=is_symmetric, has_heavy_tails=has_heavy_tails,
            exponential_cv_score=exponential_cv_score,
            uniform_variance_score=uniform_variance_score,
            lognormal_score=lognormal_score,
            l_cv=l_cv_val, l_skew=l_skew_val, l_kurtosis=l_kurt_val,
            hazard_slope=hazard_slope_val
        )


@dataclass
class TransformInfo:
    """
    Stores information about data transformations (shifting, scaling) applied
    prior to fitting a distribution. Used to reverse-transform fitted parameters.
    """
    shifted: bool = False      # Was data shifted (e.g., to ensure positivity)?
    scaled: bool = False       # Was data scaled (e.g., to [0,1] for Beta)?
    shift_value: float = 0.0   # The value added (if shifted) or original min (if scaled to [0,1])
    scale_factor: float = 1.0  # The factor by which data was divided (if scaled) or original range

    def reverse_transform_params(self, params: Tuple, dist_name: str) -> Tuple:
        """
        Adjusts fitted distribution parameters back to the original data's scale and location.
        Scipy parameter order is typically (shape_args..., loc, scale).
        """
        if not self.shifted and not self.scaled:
            return params # No transformation was applied

        params_list = list(params)
        if len(params_list) < 2:
            return params

        # SciPy continuous distributions use (shape(s)..., loc, scale).
        loc_idx = -2
        scale_idx = -1

        # Reverse scaling: data_prepared = (data - shift_value) / scale_factor
        if self.scaled:
            params_list[loc_idx] = params_list[loc_idx] * self.scale_factor + self.shift_value
            params_list[scale_idx] = params_list[scale_idx] * self.scale_factor

        # Reverse shifting: data_prepared = data + shift_value (shift_value > 0).
        if self.shifted:
            params_list[loc_idx] = params_list[loc_idx] - self.shift_value

        return tuple(params_list)


@dataclass
class LRTResult:
    """
    Stores results of Likelihood Ratio Test between nested distributions.
    """
    simple_dist: str          # Name of the simpler (nested) distribution
    complex_dist: str         # Name of the more complex distribution
    lrt_statistic: float      # -2 * (loglik_simple - loglik_complex)
    p_value: float            # P-value from chi-squared test
    df: int                   # Degrees of freedom (difference in parameter count)
    prefer_complex: bool      # True if complex model is significantly better
    simple_loglik: float      # Log-likelihood of simple model
    complex_loglik: float     # Log-likelihood of complex model


@dataclass
class CVResult:
    """
    Stores cross-validation results for a distribution.
    """
    dist_name: str
    cv_loglik_mean: float     # Mean log-likelihood from CV
    cv_loglik_std: float      # Standard deviation of CV log-likelihoods
    cv_score: float           # Composite CV score (higher is better)
    successful_folds: int     # Number of successful CV folds
    total_folds: int          # Total number of CV folds attempted
    # Optional: store per-fold log-likelihoods for oracle diagnostics (tail-risk, etc.).
    cv_logliks: Optional[List[float]] = None


@dataclass
class FitResult:
    """
    Holds the results of fitting a single distribution to data, including parameters,
    goodness-of-fit statistics, and any applied transformations.
    """
    name: str                       # Name of the distribution (e.g., "norm")
    params: Optional[Tuple]         # Fitted parameters (on original data scale)
    transform_info: TransformInfo   # Information about pre-fitting data transformations
    
    # Goodness-of-fit statistics
    ks_stat: float = float('inf')   # Kolmogorov-Smirnov statistic
    ks_p: float = 0.0               # Kolmogorov-Smirnov p-value
    cvm_stat: float = float('inf')  # Cramer-von Mises statistic
    cvm_p: float = 0.0              # Cramer-von Mises p-value
    ad_stat: Optional[float] = None # Anderson-Darling statistic (often context-dependent)
    ad_p: Optional[float] = None    # Bootstrap-calibrated AD p-value when enabled
    qq_r2: Optional[float] = None   # Optional Q-Q fit summary (squared correlation)
    pp_r2: Optional[float] = None   # Optional P-P fit summary (squared correlation)
    pp_mae: Optional[float] = None  # Optional P-P mean absolute error
    crps: Optional[float] = None    # Optional CRPS estimate (lower is better)
    
    # Information criteria and log-likelihood
    aic: float = float('inf')       # Akaike Information Criterion
    aicc: float = float('inf')      # Corrected AIC
    bic: float = float('inf')       # Bayesian Information Criterion
    loglik: float = -float('inf')   # Log-likelihood
    
    # Cross-validation results
    cv_result: Optional[CVResult] = None  # Cross-validation results
    preq_loglik_mean: Optional[float] = None  # Optional prequential/holdout predictive log-likelihood (higher is better)
    
    success: bool = False           # Was the fitting process successful?
    error: Optional[str] = None     # Error message if fitting failed
    feature_bonus: float = 0.0      # Bonus score based on data features (0 to 1)
    fit_method: Optional[str] = None # Which fitting method produced params (MLE/MPS/MoM/DE)
    mnpo_weight: Optional[float] = None  # MNPO oracle equilibrium weight (criterion=mnpo_oracle)
    
    @property
    def simple_score(self) -> float:
        """
        A composite score to rank distributions (lower is better).
        Combines p-values from GOF tests, feature bonus, and CV results.
        """
        if not self.success:
            return float('inf') # Unsuccessful fits get the worst score
        
        # Base score from GOF tests and feature bonus
        base_score = 0.6 * (1.0 - self.cvm_p) + \
                     0.3 * (1.0 - self.ks_p) - \
                     0.1 * self.feature_bonus 
        
        # Add CV component if available (weight 0.2, reduce base weights proportionally)
        if self.cv_result is not None and self.cv_result.successful_folds > 0:
            # Adjust base score weight to 0.8 and add CV component with weight 0.2
            cv_penalty = 0.2 * (1.0 - min(1.0, max(0.0, self.cv_result.cv_score)))
            base_score = 0.8 * base_score + cv_penalty
        
        return base_score


class UnifiedDistributionSelectorV6:
    """
    Hybrid distribution selector combining statistical fitting, goodness-of-fit tests,
    and feature-based heuristics to identify the best-fitting distribution for given data.
    This version incorporates improvements for robustness and accuracy, including LRT and CV.
    """
    
    def __init__(self, 
                 distributions: Optional[Dict[str, sps.rv_continuous]] = None,
                 robust_mode: bool = True,          # Enable robust outlier handling?
                 use_adaptive_strategy: bool = True, # Use feature-based adaptive strategies?
                 use_lrt: bool = True,              # Enable Likelihood Ratio Tests?
                 use_cv: bool = True,               # Enable Cross-Validation?
                 n_jobs: int = 1,                   # Number of parallel jobs (-1 for all cores)
                 # Optional prescreening (opt-in): limit fitted candidate count using L-moment ratios.
                 use_lmoment_prescreen: bool = False,
                 lmoment_prescreen_max_candidates: int = 0,
                 # Estimator for parameters (opt-in; default preserves baseline behavior).
                 fit_estimator: str = "mle",        # mle | mps
                 mps_maxiter: int = 250,
                 mps_tol: float = 1e-6,
                 # Diagnostics / additional GOF (opt-in).
                 compute_ad: bool = False,          # Compute Anderson-Darling statistic (and optional bootstrap p-value).
                 ad_bootstrap_samples: int = 0,     # If >0 and compute_ad, estimate p-value via parametric bootstrap.
                 compute_qq_pp: bool = False,       # Compute lightweight Q-Q / P-P summary metrics.
                 # Advanced DF: interval likelihood for heaped/rounded data (opt-in).
                 interval_likelihood: bool = False,
                 interval_delta: float = 0.0,
                 # Proper scoring rule diagnostics (opt-in).
                 compute_crps: bool = False,        # Compute (MC-estimated) CRPS for fitted distributions.
                 crps_mc_samples: int = 96,         # MC samples per family for CRPS estimate.
                 crps_data_subsample: int = 256,    # Max n used from data for CRPS estimate.
                 random_state: Optional[int] = None,# Seed for bootstrap diagnostics (does not affect MLE determinism).
                 # MNPO-style multi-oracle aggregation (opt-in via criterion="mnpo_oracle").
                 mnpo_use_tritrust: bool = True,
                 mnpo_include_crps: bool = False,
                 mnpo_include_preq: bool = False,
                 mnpo_use_tail_risk_oracle: bool = False,
                 mnpo_tail_risk_alpha: float = 0.33,
                 mnpo_use_qre_smoothing: bool = False,
                 mnpo_qre_temperature_gamma: float = 1.0,
                 mnpo_use_oracle_redundancy_penalty: bool = False,
                 mnpo_compute_tremble_sensitivity: bool = False,
                 preq_holdout_fraction: float = 0.20,
                 preq_min_train: int = 20,
                 preq_max_test_points: int = 128,
                 mnpo_mirror_descent_steps: int = 120,
                 mnpo_mirror_descent_eta: float = 0.18,
                 mnpo_mirror_descent_lambda: float = 0.08,
                 mnpo_pairwise_tie_margin: float = 0.02,
                 ):
        self.distributions = distributions or self._get_default_distributions()
        self.robust_mode = robust_mode
        self.use_adaptive_strategy = use_adaptive_strategy
        self.use_lrt = use_lrt
        self.use_cv = use_cv
        self.use_lmoment_prescreen = bool(use_lmoment_prescreen)
        self.lmoment_prescreen_max_candidates = int(max(0, lmoment_prescreen_max_candidates))
        estimator = str(fit_estimator or "mle").strip().lower()
        if estimator not in {"mle", "mps"}:
            estimator = "mle"
        self.fit_estimator = estimator
        self.mps_maxiter = int(max(10, mps_maxiter))
        self.mps_tol = float(max(0.0, mps_tol))
        self.compute_ad = bool(compute_ad)
        self.ad_bootstrap_samples = int(max(0, ad_bootstrap_samples))
        self.compute_qq_pp = bool(compute_qq_pp)
        self.interval_likelihood = bool(interval_likelihood)
        self.interval_delta = float(max(0.0, interval_delta))
        self.compute_crps = bool(compute_crps)
        self.crps_mc_samples = int(max(0, crps_mc_samples))
        self.crps_data_subsample = int(max(0, crps_data_subsample))
        self.random_state = None if random_state is None else int(random_state)
        self.mnpo_use_tritrust = bool(mnpo_use_tritrust)
        self.mnpo_include_crps = bool(mnpo_include_crps)
        self.mnpo_include_preq = bool(mnpo_include_preq)
        # T-DS3: tail-risk oracle removed; keep args as no-op compatibility.
        self.mnpo_use_tail_risk_oracle = False
        self.mnpo_tail_risk_alpha = float(np.clip(mnpo_tail_risk_alpha, 0.01, 1.0))
        self.mnpo_use_qre_smoothing = bool(mnpo_use_qre_smoothing)
        self.mnpo_qre_temperature_gamma = float(max(1e-6, mnpo_qre_temperature_gamma))
        self.mnpo_use_oracle_redundancy_penalty = bool(mnpo_use_oracle_redundancy_penalty)
        self.mnpo_compute_tremble_sensitivity = bool(mnpo_compute_tremble_sensitivity)
        self.preq_holdout_fraction = float(np.clip(preq_holdout_fraction, 0.01, 0.80))
        self.preq_min_train = int(max(3, preq_min_train))
        self.preq_max_test_points = int(max(1, preq_max_test_points))
        self.mnpo_mirror_descent_steps = int(max(1, mnpo_mirror_descent_steps))
        self.mnpo_mirror_descent_eta = float(mnpo_mirror_descent_eta)
        self.mnpo_mirror_descent_lambda = float(mnpo_mirror_descent_lambda)
        self.mnpo_pairwise_tie_margin = float(max(1e-4, mnpo_pairwise_tie_margin))
        # Optional debug/attribution payload emitted when criterion="mnpo_oracle".
        self.mnpo_diagnostics_: Dict[str, Any] = {}
        
        # Set up parallel processing
        if n_jobs == -1:
            self.n_jobs = os.cpu_count()
        elif n_jobs <= 0:
            self.n_jobs = 1
        else:
            self.n_jobs = min(n_jobs, len(self.distributions))
        
        self._init_distribution_strategies() # Initialize fitting/outlier strategies per distribution
        self._init_nested_models() # Initialize nested model relationships for LRT
        
    @staticmethod
    def _get_default_distributions() -> Dict[str, sps.rv_continuous]:
        """Returns a dictionary of default SciPy continuous distribution objects."""
        return {
            # Basic distributions
            "norm": sps.norm,
            "expon": sps.expon,
            "uniform": sps.uniform,
            
            # Flexible shape distributions
            "weibull_min": sps.weibull_min,
            "gamma": sps.gamma,
            "lognorm": sps.lognorm,
            "beta": sps.beta,
            
            # Heavy-tailed distributions
            "t": sps.t,
            "laplace": sps.laplace,
            "pareto": sps.pareto,
            
            # Extreme value distributions
            "gumbel_l": sps.gumbel_l,
            "gumbel_r": sps.gumbel_r,
            
            # Skewed distributions
            "powerlaw": sps.powerlaw,
            "triang": sps.triang,
            
            # Johnson system (very flexible)
            "johnsonsu": sps.johnsonsu,
            "johnsonsb": sps.johnsonsb,
        }

    @staticmethod
    def _get_extended_distributions() -> Dict[str, sps.rv_continuous]:
        """Opt-in expanded distribution library (keeps the V6 base set intact).

        This is intentionally not the default candidate set to preserve baseline behavior
        and runtime characteristics. Consumers must opt in explicitly.
        """
        base = dict(UnifiedDistributionSelectorV6._get_default_distributions())
        # Real-line: skew-normal, generalized extreme value.
        base.update(
            {
                "skewnorm": sps.skewnorm,
                "genextreme": sps.genextreme,
            }
        )
        # Positive support: inverse-gamma, log-logistic (Fisk), generalized Pareto, generalized gamma.
        base.update(
            {
                "invgamma": sps.invgamma,
                "fisk": sps.fisk,
                "genpareto": sps.genpareto,
                "gengamma": sps.gengamma,
            }
        )
        return base

    @staticmethod
    def _get_flex_distributions() -> Dict[str, sps.rv_continuous]:
        """Opt-in flexible fallback library.

        This extends the `extended` set with additional SciPy families that can
        improve fit coverage on non-ideal real-world features, at the cost of
        additional fitting time and occasional optimizer brittleness.
        """
        base = dict(UnifiedDistributionSelectorV6._get_extended_distributions())
        # Real-line flexible families.
        base.update(
            {
                "tukeylambda": sps.tukeylambda,
                "gennorm": sps.gennorm,
                "genlogistic": sps.genlogistic,
                "logistic": sps.logistic,
                "moyal": sps.moyal,
                "genhyperbolic": sps.genhyperbolic,
            }
        )
        # Positive-support families.
        base.update(
            {
                "invweibull": sps.invweibull,
                "invgauss": sps.invgauss,
                "geninvgauss": sps.geninvgauss,
            }
        )
        return base
    
    def _init_distribution_strategies(self):
        """
        Initializes strategies for each distribution, including outlier removal methods,
        parameters for those methods, and primary tests or characteristics to focus on.
        """
        self.strategies = {
            # Basic distributions
            "norm": {"outlier_method": "iqr", "iqr_multiplier": 2.5, "min_retained": 0.90},
            "expon": {"outlier_method": "lower_only", "min_retained": 0.90},
            "uniform": {"outlier_method": "strict", "iqr_multiplier": 3.5, "min_retained": 0.98},
            
            # Flexible shape distributions
            "gamma": {"outlier_method": "upper_tail", "tail_percentile": 99.0, "min_retained": 0.95},
            "lognorm": {"outlier_method": "conservative", "iqr_multiplier": 3.5, "min_retained": 0.95},
            "beta": {"outlier_method": "iqr", "iqr_multiplier": 2.5, "min_retained": 0.90},
            "weibull_min": {"outlier_method": "bypass"},
            
            # Heavy-tailed distributions
            "t": {"outlier_method": "iqr", "iqr_multiplier": 3.0, "min_retained": 0.95},
            "laplace": {"outlier_method": "iqr", "iqr_multiplier": 2.0, "min_retained": 0.92},
            "pareto": {"outlier_method": "upper_tail", "tail_percentile": 98.0, "min_retained": 0.90},
            
            # Extreme value distributions
            "gumbel_l": {"outlier_method": "iqr", "iqr_multiplier": 2.5, "min_retained": 0.90},
            "gumbel_r": {"outlier_method": "iqr", "iqr_multiplier": 2.5, "min_retained": 0.90},
            
            # Skewed distributions
            "powerlaw": {"outlier_method": "upper_tail", "tail_percentile": 99.0, "min_retained": 0.95},
            "triang": {"outlier_method": "iqr", "iqr_multiplier": 2.5, "min_retained": 0.90},
            
            # Johnson system
            "johnsonsu": {"outlier_method": "iqr", "iqr_multiplier": 2.0, "min_retained": 0.90},
            "johnsonsb": {"outlier_method": "iqr", "iqr_multiplier": 2.0, "min_retained": 0.90},
        }

    def _init_nested_models(self):
        """
        Initialize nested model relationships for Likelihood Ratio Tests.
        Format: {complex_model: [(simple_model, constraint_description), ...]}
        """
        self.nested_models = {
            # Shape parameter relationships
            "weibull_min": [("expon", "shape parameter = 1")],
            "gamma": [("expon", "shape parameter = 1")],
            
            # Heavy-tail relationships
            "t": [("norm", "degrees of freedom → ∞"), ("laplace", "df = 1")],
            "laplace": [("norm", "scale parameter ratio = √2")],
            
            # Johnson system can approximate many distributions
            "johnsonsu": [("norm", "shape parameters → specific values")],
            
            # Triangular can reduce to uniform
            "triang": [("uniform", "shape parameter = 0.5")],
            
            # Power law relationships
            "pareto": [("powerlaw", "different parameterizations")],
        }

    def _select_outlier_strategy(self, dist_name: str, features: Optional[DistributionFeatures]) -> str:
        """
        Selects an outlier removal strategy for a given distribution.
        Can be adaptive based on data features if `use_adaptive_strategy` is True.
        """
        # Default strategy from pre-defined settings for the distribution
        default_strategy = self.strategies.get(dist_name, {}).get("outlier_method", "iqr")

        if not self.use_adaptive_strategy or features is None:
            return default_strategy

        # Example adaptive overrides (can be expanded)
        if dist_name == "expon" and features.exponential_cv_score > 0.8:
            return "lower_only" # Reinforce for exponential
        if dist_name == "uniform" and features.uniform_variance_score > 0.7:
            return "strict" # Reinforce for uniform
        
        return default_strategy # Fallback to the distribution's default
    
    def _apply_outlier_removal(self, data: np.ndarray, dist_name: str, 
                             features: Optional[DistributionFeatures]) -> np.ndarray:
        """
        Applies outlier removal to the data based on the selected strategy for `dist_name`.
        Only active if `self.robust_mode` is True and data is sufficient.
        """
        if not self.robust_mode or len(data) < 20: # Minimum data size for outlier removal
            return data

        strategy_name = self._select_outlier_strategy(dist_name, features)
        dist_config = self.strategies.get(dist_name, {})
        
        # MODIFICATION: For Weibull_min, robust outlier removal was detrimental.
        # The "bypass" strategy effectively skips removal here.
        if strategy_name == "bypass":
            return data

        original_len = len(data)
        # Ensure a minimum percentage of data is retained after removal
        min_retained_abs = int(original_len * dist_config.get("min_retained", 0.90))
        
        processed_data = data # Start with original data

        if strategy_name == "conservative" or strategy_name == "strict" or strategy_name == "iqr":
            q1, q3 = np.percentile(data, [25, 75])
            iqr_val = q3 - q1
            if iqr_val > 1e-9: # Proceed only if IQR is meaningful
                multiplier = dist_config.get("iqr_multiplier", 2.5 if strategy_name == "iqr" else 3.5)
                lower_bound = q1 - multiplier * iqr_val
                upper_bound = q3 + multiplier * iqr_val
                mask = (data >= lower_bound) & (data <= upper_bound)
                if np.sum(mask) >= min_retained_abs:
                    processed_data = data[mask]
        
        elif strategy_name == "lower_only": # For exponential-like distributions
            mask = data > 1e-9 # Remove non-strictly positive values
            if np.sum(mask) >= min_retained_abs and np.sum(mask) > 0:
                 processed_data = data[mask]
            elif np.sum(data > 0) > 0: # Fallback: if too much removed, just take any positive
                processed_data = data[data > 0]

        elif strategy_name == "upper_tail": # For right-skewed distributions like Gamma
            percentile_thresh = dist_config.get("tail_percentile", 99.0)
            upper_bound_val = np.percentile(data, percentile_thresh)
            mask = data <= upper_bound_val
            if np.sum(mask) >= min_retained_abs:
                processed_data = data[mask]
        
        return processed_data
    
    def _calculate_feature_bonus(self, features: Optional[DistributionFeatures], dist_name: str) -> float:
        """
        Calculates a bonus score (0-1) based on how well data features match
        the expected characteristics of `dist_name`. Higher bonus is better.
        Enhanced with support for new distributions.
        """
        if features is None: return 0.0

        if dist_name in _POSITIVE_ONLY_FAMILIES:
            frac_negative = float(getattr(features, "frac_negative", 0.0) or 0.0)
            # Positive-support bonuses must not fire on genuinely real-valued data.
            if (not bool(getattr(features, "is_positive", False))) or frac_negative > 0.0:
                return 0.0
        
        bonus = 0.0
        if dist_name == "expon":
            if features.exponential_cv_score > 0.85: bonus = features.exponential_cv_score
            if features.exponential_cv_score > 0.95: bonus = max(bonus, features.exponential_cv_score * 1.1)
            if abs(features.l_cv - 1.0) < 0.1: bonus = max(bonus, 0.9)
        elif dist_name == "uniform":
            if features.uniform_variance_score > 0.8: bonus = features.uniform_variance_score
        elif dist_name == "lognorm":
            if features.lognormal_score > 0.05 and features.skewness > 0.5:
                bonus = min(features.lognormal_score * 2.0, 1.0)
                if features.skewness > 1.0: bonus = min(bonus + 0.2, 1.0)
        elif dist_name == "norm":
            if features.is_symmetric and not features.has_heavy_tails: bonus = 0.5
            if abs(features.skewness) < 0.2 and abs(features.excess_kurtosis) < 0.5: bonus = 0.8
            if abs(features.l_skew) < 0.1 and abs(features.l_kurtosis) < 0.2: bonus = max(bonus, 0.9)
        elif dist_name == "gamma":
            if features.is_positive and 0.5 < features.skewness < 3.0: bonus = 0.3
            if 0.3 < features.cv < 1.5: bonus = max(bonus, 0.6)
            if features.exponential_cv_score > 0.95 and features.is_positive: bonus = max(0.0, bonus - 0.2)
            if features.hazard_slope > 1.2: bonus = max(bonus, 0.7)
        elif dist_name == "beta":
            if abs(features.skewness) < 1.5 and features.excess_kurtosis < 1.0: bonus = 0.4
            if features.uniform_variance_score > 0.85: bonus = max(0.0, bonus - 0.25)
        elif dist_name == "weibull_min":
            if features.is_positive:
                if 0 < features.skewness < 2.5: bonus = 0.4
                elif features.skewness <= 0 and features.excess_kurtosis > 0: bonus = 0.3
                if features.hazard_slope < 1.0: bonus = max(bonus, 0.6)
        # New distributions
        elif dist_name == "t":
            if features.has_heavy_tails and features.is_symmetric: bonus = 0.6
            if abs(features.skewness) < 0.3 and features.excess_kurtosis > 1.0: bonus = max(bonus, 0.7)
        elif dist_name == "laplace":
            if features.is_symmetric and features.excess_kurtosis > 0.5: bonus = 0.5
            if abs(features.skewness) < 0.2 and features.excess_kurtosis > 1.5: bonus = 0.8
        elif dist_name == "pareto":
            if features.is_positive and features.skewness > 2.0: bonus = 0.6
            if features.cv > 1.5: bonus = max(bonus, 0.7)
        elif dist_name in ["gumbel_l", "gumbel_r"]:
            if 0.5 < abs(features.skewness) < 2.0: bonus = 0.4
            if features.excess_kurtosis > 0.5: bonus = max(bonus, 0.5)
        elif dist_name == "powerlaw":
            if features.is_positive and features.skewness > 1.0: bonus = 0.5
        elif dist_name == "triang":
            if abs(features.skewness) < 1.0 and -1.0 < features.excess_kurtosis < 1.0: bonus = 0.4
        elif dist_name in ["johnsonsu", "johnsonsb"]:
            # Johnson distributions are very flexible, give modest bonus for any reasonable shape
            bonus = 0.3
            if abs(features.skewness) < 2.0 and abs(features.excess_kurtosis) < 5.0: bonus = 0.4
        
        return max(0.0, min(1.0, bonus))

    def _lmoment_prescreen_distribution_names(
        self,
        dist_names: Sequence[str],
        features: DistributionFeatures,
    ) -> List[str]:
        """Limit candidate distributions using L-moment ratios (opt-in).

        This is intentionally conservative: it ranks candidates by simple
        heuristics driven by L-skewness/L-kurtosis and keeps the top-K.
        """
        names = [str(n) for n in dist_names]
        max_k = int(getattr(self, "lmoment_prescreen_max_candidates", 0) or 0)
        if max_k <= 0 or len(names) <= max_k:
            return names

        l_skew = float(getattr(features, "l_skew", 0.0) or 0.0)
        l_kurt = float(getattr(features, "l_kurtosis", 0.0) or 0.0)
        l_cv = float(getattr(features, "l_cv", 0.0) or 0.0)
        is_pos = bool(getattr(features, "is_positive", False))
        frac_negative = float(getattr(features, "frac_negative", 0.0) or 0.0)

        positive_like = set(_POSITIVE_ONLY_FAMILIES)
        symmetric_like = {
            "norm",
            "t",
            "laplace",
            "logistic",
            "moyal",
            "gennorm",
            "genhyperbolic",
        }
        bounded_like = {"uniform", "beta", "powerlaw", "triang", "johnsonsb"}
        heavy_tail_like = {"t", "laplace", "pareto", "genpareto", "genhyperbolic", "johnsonsu"}

        mandatory = {"norm", "t", "johnsonsu", "gamma", "lognorm", "weibull_min"}

        if (not is_pos) and frac_negative >= 0.05:
            names = [n for n in names if n not in positive_like]
            if len(names) <= max_k:
                return names

        def score(name: str) -> float:
            s = float(self._calculate_feature_bonus(features, name))

            # Prefer positive-support families when data is strictly positive.
            if name in positive_like:
                if is_pos:
                    s += 0.25
                else:
                    # Strongly demote positive-only families for real-valued data.
                    s += -0.80
                    if frac_negative >= 0.05:
                        s += -0.10
                s += 0.15 * float(np.clip((l_skew - 0.12) / 0.35, 0.0, 1.0))
                s += 0.10 * float(np.clip((l_cv - 0.35) / 1.2, 0.0, 1.0))

            # Prefer symmetric families when L-skewness is small.
            if name in symmetric_like:
                s += 0.20 * float(np.clip((0.22 - abs(l_skew)) / 0.22, 0.0, 1.0))

            # Prefer heavy-tail families when L-kurtosis is large.
            if name in heavy_tail_like:
                s += 0.20 * float(np.clip((l_kurt - 0.13) / 0.20, 0.0, 1.0))

            # Prefer bounded families when L-kurtosis is very small (uniform-ish).
            if name in bounded_like:
                s += 0.15 * float(np.clip((0.07 - l_kurt) / 0.07, 0.0, 1.0))

            # Mild bias for a small safety subset.
            if name in mandatory:
                s += 0.10

            return float(s)

        ranked = sorted(names, key=score, reverse=True)

        # Ensure we keep at least one reasonable candidate even if ranking is odd.
        keep = ranked[: max(1, min(int(max_k), len(ranked)))]
        return keep
    
    def _prepare_data(self, data: np.ndarray, dist_name: str, 
                     features: Optional[DistributionFeatures]) -> Tuple[np.ndarray, TransformInfo]:
        """
        Prepares data for fitting: applies outlier removal (via _apply_outlier_removal)
        and then applies transformations (shifting/scaling) to meet distribution support requirements.
        """
        data_arr = np.asarray(data) # Ensure it's a numpy array
        # data_clean_nan is already handled before calling this in the flow.
        # Here, `data` is assumed to be NaN-cleaned.
        
        transform_info = TransformInfo()
        
        # Apply robust outlier removal first (if enabled and applicable)
        data_after_outliers = self._apply_outlier_removal(data_arr.copy(), dist_name, features)
        
        # Use data after outlier removal for support transformations.
        # If outlier removal was too aggressive and removed all data, fallback to pre-outlier data.
        data_to_process = data_after_outliers if len(data_after_outliers) > 0 else data_arr
        if len(data_to_process) == 0: return data_to_process, transform_info # Should not happen if initial checks pass

        # Apply support transformations (shifting/scaling)
        if dist_name in [
            "gamma",
            "lognorm",
            "weibull_min",
            "expon",
            "pareto",
            # Extended positive-support families (opt-in via distribution library).
            "invgamma",
            "fisk",
            "genpareto",
            "gengamma",
            # Flex positive-support families (opt-in via distribution library).
            "invweibull",
            "invgauss",
            "geninvgauss",
        ]:
            min_val = np.min(data_to_process)
            if min_val <= 1e-9: # If data is not strictly positive
                std_dev = np.std(data_to_process)
                # Shift by a small positive amount
                shift_amount = abs(min_val) + (std_dev * 0.01 if std_dev > 1e-6 else 0.01)
                shift_amount = max(shift_amount, 1e-9) # Ensure shift is positive
                if std_dev > 1e-9 and shift_amount > 2.0 * float(std_dev):
                    logger.warning(
                        "Large positive-support shift applied for %s: shift=%.6g std=%.6g",
                        dist_name,
                        float(shift_amount),
                        float(std_dev),
                    )
                data_to_process = data_to_process + shift_amount
                transform_info.shifted = True
                transform_info.shift_value = shift_amount # Store the positive amount added
                
        elif dist_name in ["beta", "powerlaw"]: # Distributions requiring [0, 1] support
            min_val, max_val = np.min(data_to_process), np.max(data_to_process)
            # Check if scaling is necessary
            if not (min_val >= -1e-9 and max_val <= 1.0 + 1e-9 and (max_val - min_val) <= 1.0 + 1e-9):
                 # Only scale if data is outside a slightly tolerant [0,1] or range is > 1
                range_val = max_val - min_val
                if range_val < 1e-9: # If all values are (nearly) the same
                    # If this constant value is outside [0,1], map it to 0.5 within [0,1]
                    if min_val < -1e-9 or max_val > 1.0 + 1e-9:
                        data_to_process = np.full_like(data_to_process, 0.5) 
                        transform_info.scaled = True
                        transform_info.shift_value = min_val    # Original min
                        transform_info.scale_factor = range_val if range_val > 1e-9 else 1.0 # Original range
                else: # Range is significant, scale to [0,1]
                    data_to_process = (data_to_process - min_val) / range_val
                    transform_info.scaled = True
                    transform_info.shift_value = min_val    # Store original min for reversing loc
                    transform_info.scale_factor = range_val # Store original range for reversing scale
        
        return data_to_process, transform_info
    
    def _fit_single_distribution(
        self,
        data_original_nan_cleaned: np.ndarray,
        dist_name: str,
        features: Optional[DistributionFeatures],
        is_cv_fold: bool = False,
        *,
        compute_gof: bool = True,
    ) -> FitResult:
        """
        Fits a single specified distribution to the data.
        It prepares data (outlier removal, support transforms), tries multiple fitting methods,
        reverses parameter transformations, and calculates goodness-of-fit.
        If is_cv_fold is True, it will not attempt to perform cross-validation again.
        If compute_gof is False, goodness-of-fit metrics are skipped (params-only fit).
        """
        try:
            dist_scipy_obj = self.distributions[dist_name]
            
            # Prepare data for fitting (applies outlier removal and support transformations)
            data_prepared_for_fitting, transform_info = self._prepare_data(
                data_original_nan_cleaned.copy(), dist_name, features
            )
            
            if len(data_prepared_for_fitting) < 10: # Minimum data for reliable fitting
                # For CV folds, this might be too strict, allow fewer samples.
                # However, the internal fit methods might still fail.
                min_samples = 3 if is_cv_fold else 10
                if len(data_prepared_for_fitting) < min_samples:
                    raise ValueError(f"Less than {min_samples} samples for {dist_name} after preprocessing.")
            
            params_fitted_on_prepared_data = None
            fit_methods_tried = []

            # Optional Method 0: Maximum Product of Spacings (opt-in via fit_estimator="mps").
            if str(getattr(self, "fit_estimator", "mle") or "mle").strip().lower() == "mps":
                try:
                    params_mps = self._fit_distribution_mps(
                        data_prepared_for_fitting, dist_scipy_obj, dist_name, features
                    )
                    if params_mps is not None and all(np.isfinite(p) for p in params_mps):
                        params_fitted_on_prepared_data = params_mps
                        fit_methods_tried.append("MPS")
                except Exception as exc:
                    params_fitted_on_prepared_data = None

            # Attempt fitting methods in order of preference/speed
            # Method 1: Standard MLE (SciPy's default) with enhanced initial guesses and constraints
            if params_fitted_on_prepared_data is None:
                try:
                    # Get enhanced fitting constraints and initial guesses
                    fit_kwargs = self._get_fit_bounds_and_constraints(data_prepared_for_fitting, dist_name, features)
                    initial_guess = self._get_initial_parameter_guess(data_prepared_for_fitting, dist_name, features)

                    # Add initial guess if available
                    if initial_guess is not None:
                        fit_kwargs['guess'] = initial_guess

                    params_fitted_on_prepared_data = dist_scipy_obj.fit(data_prepared_for_fitting, **fit_kwargs)
                    if not all(np.isfinite(p) for p in params_fitted_on_prepared_data):
                        params_fitted_on_prepared_data = None
                    else:
                        fit_methods_tried.append("MLE_enhanced")
                except Exception as exc:
                    params_fitted_on_prepared_data = None  # Enhanced MLE Failed
            
            # Method 2: Fallback to basic MLE if enhanced version failed
            if params_fitted_on_prepared_data is None:
                try:
                    basic_kwargs = {}
                    if dist_name == "beta" and transform_info.scaled:
                        basic_kwargs['floc'] = 0
                        basic_kwargs['fscale'] = 1
                    
                    params_fitted_on_prepared_data = dist_scipy_obj.fit(data_prepared_for_fitting, **basic_kwargs)
                    if not all(np.isfinite(p) for p in params_fitted_on_prepared_data): 
                        params_fitted_on_prepared_data = None
                    else: 
                        fit_methods_tried.append("MLE_basic")
                except Exception as exc: params_fitted_on_prepared_data = None

            # Method 3: Method of Moments (if MLE failed, for specific distributions)
            if params_fitted_on_prepared_data is None and dist_name in ["norm", "expon", "gamma"]:
                try:
                    params_fitted_on_prepared_data = self._fit_method_of_moments(data_prepared_for_fitting, dist_name)
                    if params_fitted_on_prepared_data is not None and not all(np.isfinite(p) for p in params_fitted_on_prepared_data):
                        params_fitted_on_prepared_data = None
                    elif params_fitted_on_prepared_data is not None:
                        fit_methods_tried.append("MoM")
                except Exception as exc: params_fitted_on_prepared_data = None # MoM Failed

            # Method 4: Differential Evolution (robust fallback if others failed)
            if params_fitted_on_prepared_data is None and dist_name in ["gamma", "weibull_min", "lognorm", "beta", "t", "pareto", "johnsonsu", "johnsonsb"]:
                try:
                    params_fitted_on_prepared_data = self._fit_distribution_differential_evolution(
                        data_prepared_for_fitting, dist_scipy_obj, dist_name
                    )
                    if params_fitted_on_prepared_data is not None and not all(np.isfinite(p) for p in params_fitted_on_prepared_data):
                        params_fitted_on_prepared_data = None
                    elif params_fitted_on_prepared_data is not None:
                        fit_methods_tried.append("DE")
                except Exception as exc: params_fitted_on_prepared_data = None # DE Failed
            
            if params_fitted_on_prepared_data is None:
                raise ValueError(f"All fitting methods ({', '.join(fit_methods_tried) or 'none'}) failed.")
            
            # Reverse parameter transformations to match original data scale
            params_on_original_scale = transform_info.reverse_transform_params(params_fitted_on_prepared_data, dist_name)
            
            gof_metrics = {}
            if bool(compute_gof):
                # Calculate goodness-of-fit using original (NaN-cleaned) data and original-scale parameters
                gof_metrics = self._calculate_goodness_of_fit(
                    data_original_nan_cleaned,
                    dist_scipy_obj,
                    params_on_original_scale,
                    dist_name,
                    compute_ad_p=not bool(is_cv_fold),
                )
            
            # Calculate feature bonus (using features from data before distribution-specific outlier removal)
            feature_bonus = self._calculate_feature_bonus(features, dist_name)
            
            # Perform cross-validation if enabled AND not already in a CV fold
            cv_result = None
            if self.use_cv and not is_cv_fold and len(data_original_nan_cleaned) >= 10:
                cv_result = self._perform_cross_validation(data_original_nan_cleaned, dist_name, features)
            
            return FitResult(
                name=dist_name, params=params_on_original_scale, transform_info=transform_info,
                success=True,
                feature_bonus=feature_bonus,
                fit_method=(fit_methods_tried[-1] if fit_methods_tried else None),
                cv_result=cv_result,
                **gof_metrics
            )
        except Exception as e: # Catch any error during the fitting process for this distribution
            return FitResult(name=dist_name, params=None, transform_info=TransformInfo(), 
                             success=False, error=f"{dist_name} fit error: {str(e)}")

    def _fit_method_of_moments(self, data: np.ndarray, dist_name: str) -> Optional[Tuple]:
        """Fits distribution using Method of Moments (on prepared/transformed data)."""
        if len(data) < 2 or np.std(data) < 1e-9: return None # Not enough data or no variance

        try:
            if dist_name == "norm": return (np.mean(data), np.std(data, ddof=1))
            elif dist_name == "expon":
                loc = np.min(data)
                scale = np.mean(data) - loc
                return (loc, scale if scale > 1e-9 else 1e-9)
            elif dist_name == "gamma": # Fit a, loc, scale
                mean, var, sk = np.mean(data), np.var(data, ddof=1), skew(data)
                if var < 1e-9: return None
                if abs(sk) > 1e-3: # Use skewness if significant
                    a = 4 / (sk**2)
                    scale_param = np.sqrt(var / a)
                    loc_param = mean - a * scale_param
                    if a > 0 and scale_param > 0: return (a, loc_param, scale_param)
                # Fallback if skew is near zero (approximates normal or shifted exponential)
                loc_param = np.min(data) # Assume loc is min
                shifted_data = data - loc_param
                mean_s, var_s = np.mean(shifted_data), np.var(shifted_data, ddof=1)
                if mean_s > 1e-9 and var_s > 1e-9:
                    a = (mean_s**2) / var_s
                    scale_param = var_s / mean_s
                    if a > 0 and scale_param > 0: return (a, loc_param, scale_param)
        except Exception as exc: pass
        return None
    
    def _fit_distribution_differential_evolution(self, data: np.ndarray, dist_obj: sps.rv_continuous,
                                               dist_name: str) -> Optional[Tuple]:
        """Fits distribution using Differential Evolution (on prepared/transformed data)."""
        try:
            def neg_loglik_de(params_de): # Objective function for DE (minimization)
                try:
                    log_likelihood = dist_obj.logpdf(data, *params_de)
                    # Handle non-finite logpdf values that can occur with bad params
                    finite_sum = np.sum(log_likelihood[np.isfinite(log_likelihood)])
                    # Penalize if many non-finite values or sum is non-finite
                    penalty = (len(log_likelihood) - len(log_likelihood[np.isfinite(log_likelihood)])) * 1e3
                    if not np.isfinite(finite_sum): return 1e12 + penalty
                    return -finite_sum + penalty
                except Exception as exc:
                    # FIX HIGH-003: Log exception instead of silent failure (T-AUDIT-001-FIX-004)
                    logger.debug(f"DE optimizer logpdf evaluation failed for {dist_name}: {exc}")
                    return 1e12  # Large penalty for any error
            
            bounds_for_de = self._get_param_bounds_for_de(dist_name, data, dist_obj.numargs)
            if not bounds_for_de: return None

            # FIX CRITICAL-003: Use self.random_state instead of hardcoded seed=42 (T-AUDIT-001-FIX-003)
            base_seed = 0 if self.random_state is None else int(self.random_state)
            de_seed = int(base_seed) ^ 0xDE000001  # Unique seed for differential evolution
            result = differential_evolution(neg_loglik_de, bounds_for_de, seed=de_seed, 
                                            maxiter=250, tol=1e-5, polish=True, workers=1) # workers=1 for reproducibility / no Loky
            
            if result.success and np.isfinite(result.fun) and result.fun < 1e11: # Check if DE converged to a reasonable value
                # Basic validation of fitted parameters (e.g., scale > 0)
                final_params = list(result.x)
                # Scale parameter is usually the last one
                if len(final_params) > 0 and final_params[-1] <= 1e-9: final_params[-1] = 1e-9
                # Shape parameters for beta (a,b) must be positive
                if dist_name == "beta" and len(final_params) >= 2: # a, b are first two of shape params
                    if final_params[0] <= 1e-9: final_params[0] = 1e-9
                    if final_params[1] <= 1e-9: final_params[1] = 1e-9
                
                # Final check if these params yield finite log-likelihood sum
                if np.isfinite(np.sum(dist_obj.logpdf(data, *final_params))):
                    return tuple(final_params)
        except Exception as exc: pass
        return None

    def _fit_distribution_mps(
        self,
        data: np.ndarray,
        dist_obj: sps.rv_continuous,
        dist_name: str,
        features: Optional[DistributionFeatures],
    ) -> Optional[Tuple]:
        """Fit distribution parameters by Maximum Product of Spacings (MPS).

        This is an opt-in alternative to MLE that can be more stable for some
        families and small samples. Falls back to None if optimization fails.
        """
        try:
            xs = np.sort(np.asarray(data, dtype=float).ravel())
            xs = xs[np.isfinite(xs)]
            if xs.size < 8:
                return None

            bounds = self._get_param_bounds_for_de(dist_name, xs, dist_obj.numargs)
            if not bounds:
                return None

            x0 = self._get_initial_parameter_guess(xs, dist_name, features)
            if x0 is None:
                try:
                    fit_kwargs = self._get_fit_bounds_and_constraints(xs, dist_name, features)
                    x0 = dist_obj.fit(xs, **fit_kwargs)
                except Exception as exc:
                    return None

            x0_arr = np.asarray(x0, dtype=float).ravel()
            if x0_arr.size != len(bounds) or not np.all(np.isfinite(x0_arr)):
                return None

            lows = np.asarray([b[0] for b in bounds], dtype=float)
            highs = np.asarray([b[1] for b in bounds], dtype=float)
            x0_arr = np.clip(x0_arr, lows, highs)

            eps = 1e-12

            def neg_log_mps(theta: np.ndarray) -> float:
                try:
                    u = np.asarray(dist_obj.cdf(xs, *theta), dtype=float).ravel()
                except Exception as exc:
                    return 1e12
                if u.size != xs.size or not np.all(np.isfinite(u)):
                    return 1e12
                u = np.clip(u, eps, 1.0 - eps)
                spacings = np.diff(np.concatenate(([0.0], u, [1.0])))
                bad = spacings <= 0.0
                if np.any(bad):
                    spacings = np.maximum(spacings, eps)
                    penalty = 100.0 * float(np.sum(bad))
                else:
                    penalty = 0.0
                val = -float(np.sum(np.log(spacings))) + penalty
                return val if np.isfinite(val) else 1e12

            res = minimize(
                neg_log_mps,
                x0=x0_arr,
                method="L-BFGS-B",
                bounds=bounds,
                tol=float(self.mps_tol) if float(self.mps_tol) > 0 else None,
                options={"maxiter": int(self.mps_maxiter)},
            )
            if not bool(getattr(res, "success", False)):
                return None
            theta = np.asarray(getattr(res, "x", None), dtype=float).ravel()
            if theta.size != len(bounds) or not np.all(np.isfinite(theta)):
                return None

            # Enforce basic scale positivity.
            if theta.size >= 1 and theta[-1] <= 1e-9:
                theta[-1] = 1e-9

            params = tuple(float(v) for v in theta.tolist())
            # Sanity check: finite log-likelihood for fitted params.
            try:
                ll = np.sum(dist_obj.logpdf(xs, *params))
                if not np.isfinite(ll):
                    return None
            except Exception as exc:
                return None
            return params
        except Exception as exc:
            return None

    def _get_param_bounds_for_de(self, dist_name: str, data: np.ndarray, num_shape_args: int) -> List[Tuple[float, float]]:
        """Provides parameter bounds for Differential Evolution, based on prepared data characteristics."""
        d_min, d_max = np.min(data), np.max(data)
        d_mean, d_std = np.mean(data), np.std(data)
        d_range = d_max - d_min if d_max > d_min else 1e-3 # Ensure d_range is positive

        # Default bounds for loc and scale, can be overridden per distribution
        # loc_bound allows DE to search around the data's min/max
        loc_bound = (d_min - d_range * 0.5, d_max + d_range * 0.5)
        # scale_bound allows DE to search for scales from very small up to data's range/std
        scale_bound = (1e-7, max(d_std * 3, d_range * 1.5, 1e-6)) # Ensure scale_bound is positive

        # Specific bounds for distributions often requiring positive data (after potential shift)
        if dist_name in ["expon", "gamma", "lognorm", "weibull_min", "pareto"]:
            # For data prepared to be positive, loc is expected near 0 or its actual min
            loc_bound = (min(d_min * 0.9, d_min - 0.1 * d_std), # Allow slightly less than min
                         max(d_min * 1.1, d_min + 0.1 * d_std)) # Allow slightly more than min
            loc_bound = (max(loc_bound[0], -d_range*0.1), min(loc_bound[1], d_range*0.1)) # Keep loc near 0 for shifted positive data
            loc_bound = (min(loc_bound[0], loc_bound[1]-1e-7), loc_bound[1]) # Ensure loc_min < loc_max

        param_bounds = []
        # Shape parameters (vary widely, common range (0.01, 50))
        for i in range(num_shape_args):
            param_bounds.append((1e-5, 60.0)) 

        param_bounds.append(loc_bound)  # Add loc bound
        param_bounds.append(scale_bound) # Add scale bound

        # Further refinements for specific distributions
        if dist_name == "beta": # a, b, loc, scale. num_shape_args = 2 (a,b)
            # If data was scaled to [0,1] for beta fitting:
            if abs(d_min - 0) < 1e-3 and abs(d_max - 1) < 1e-3:
                param_bounds = [(1e-5, 60.0), (1e-5, 60.0), (-0.01, 0.01), (0.99, 1.01)] # Tight loc/scale for [0,1] data
        elif dist_name == "lognorm": # s (shape), loc, scale. num_shape_args = 1 (s)
            param_bounds = [(1e-5, 15.0), loc_bound, scale_bound] # Shape 's' for lognorm often smaller
        elif dist_name == "t":  # df, loc, scale
            param_bounds = [(1.1, 100.0), loc_bound, scale_bound]
        elif dist_name == "pareto":  # b (shape), loc, scale
            param_bounds = [(1.01, 20.0), loc_bound, scale_bound]  # Shape must be > 1
        elif dist_name == "powerlaw":  # a (shape), loc, scale
            param_bounds = [(1e-5, 10.0), (0, 1), (0.1, 10.0)]  # Powerlaw typically on [0,1]
        elif dist_name == "triang":  # c (shape), loc, scale
            param_bounds = [(0.001, 0.999), loc_bound, scale_bound]  # Shape c in (0,1)
        elif dist_name == "laplace":  # No shape params, just loc, scale
            param_bounds = [loc_bound, scale_bound]
        elif dist_name in ["gumbel_l", "gumbel_r"]:  # No shape params, just loc, scale
            param_bounds = [loc_bound, scale_bound]
        elif dist_name == "johnsonsu":  # a, b, loc, scale
            param_bounds = [(-10.0, 10.0), (1e-5, 10.0), loc_bound, scale_bound]
        elif dist_name == "johnsonsb":  # a, b, loc, scale
            param_bounds = [(-10.0, 10.0), (1e-5, 10.0), loc_bound, scale_bound]
        
        # Ensure all lower bounds are less than upper bounds
        for i in range(len(param_bounds)):
            low, high = param_bounds[i]
            if low >= high: param_bounds[i] = (high - 1e-7 if high > 1e-7 else 0, high)

        return param_bounds

    @staticmethod
    def _anderson_darling_stat_from_cdf(cdf_values: np.ndarray) -> float:
        """Compute Anderson-Darling statistic given CDF values at sorted samples."""
        u = np.asarray(cdf_values, dtype=float).ravel()
        n = int(u.size)
        if n <= 0:
            return float("nan")
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        i = np.arange(1, n + 1, dtype=float)
        s = np.sum((2.0 * i - 1.0) * (np.log(u) + np.log(1.0 - u[::-1])))
        stat = -float(n) - (1.0 / float(n)) * float(s)
        return float(stat) if np.isfinite(stat) else float("nan")

    def _bootstrap_ad_pvalue(
        self,
        *,
        dist_obj: sps.rv_continuous,
        params: Tuple,
        n: int,
        ad_stat_obs: float,
        dist_name: str,
    ) -> Optional[float]:
        if not (self.compute_ad and self.ad_bootstrap_samples > 0):
            return None
        if not (np.isfinite(ad_stat_obs) and int(n) >= 5):
            return None

        base_seed = 0 if self.random_state is None else int(self.random_state)
        fam_seed = int(zlib.crc32(dist_name.encode("utf-8")) & 0xFFFFFFFF)
        rng = np.random.default_rng(int(base_seed) ^ fam_seed)

        stats: List[float] = []
        for _ in range(int(self.ad_bootstrap_samples)):
            try:
                sample = dist_obj.rvs(*params, size=int(n), random_state=rng)
            except Exception as exc:
                continue
            arr = np.asarray(sample, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size < 5:
                continue
            try:
                xs = np.sort(arr)
                u = np.asarray(dist_obj.cdf(xs, *params), dtype=float)
            except Exception as exc:
                continue
            stat = self._anderson_darling_stat_from_cdf(u)
            if np.isfinite(stat):
                stats.append(float(stat))

        if not stats:
            return None

        ge = float(np.mean(np.asarray(stats, dtype=float) >= float(ad_stat_obs)))
        p = (ge * len(stats) + 1.0) / (len(stats) + 2.0)
        return float(np.clip(p, 0.0, 1.0))

    @staticmethod
    def _qq_pp_diagnostics(
        *,
        data_sorted: np.ndarray,
        dist_obj: sps.rv_continuous,
        params: Tuple,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        n = int(data_sorted.size)
        if n < 5:
            return None, None, None
        probs = (np.arange(1, n + 1, dtype=float) - 0.5) / float(n)
        try:
            q_theory = np.asarray(dist_obj.ppf(probs, *params), dtype=float)
        except Exception as exc:
            q_theory = np.full(n, np.nan, dtype=float)
        try:
            u = np.asarray(dist_obj.cdf(data_sorted, *params), dtype=float)
        except Exception as exc:
            u = np.full(n, np.nan, dtype=float)

        qq_r2 = None
        pp_r2 = None
        pp_mae = None

        mask_qq = np.isfinite(q_theory) & np.isfinite(data_sorted)
        if int(np.sum(mask_qq)) >= 5:
            x = data_sorted[mask_qq]
            y = q_theory[mask_qq]
            if float(np.std(x)) > 1e-12 and float(np.std(y)) > 1e-12:
                r = float(np.corrcoef(x, y)[0, 1])
                if np.isfinite(r):
                    qq_r2 = float(np.clip(r * r, 0.0, 1.0))

        mask_pp = np.isfinite(u)
        if int(np.sum(mask_pp)) >= 5:
            u2 = np.clip(u[mask_pp], 1e-12, 1.0 - 1e-12)
            p2 = probs[mask_pp]
            if float(np.std(u2)) > 1e-12 and float(np.std(p2)) > 1e-12:
                r = float(np.corrcoef(u2, p2)[0, 1])
                if np.isfinite(r):
                    pp_r2 = float(np.clip(r * r, 0.0, 1.0))
            pp_mae = float(np.mean(np.abs(u2 - p2)))

        return qq_r2, pp_r2, pp_mae

    def _compute_crps_mc(
        self,
        *,
        data: np.ndarray,
        dist_obj: sps.rv_continuous,
        params: Tuple,
        dist_name: str,
    ) -> Optional[float]:
        """Monte Carlo CRPS estimate for arbitrary SciPy distributions.

        Uses the identity: CRPS(F, x) = E|X - x| - 0.5 E|X - X'|.
        """
        if not bool(getattr(self, "compute_crps", False)):
            return None
        m = int(getattr(self, "crps_mc_samples", 0) or 0)
        if m <= 0:
            return None

        arr = np.asarray(data, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size < 5:
            return None

        max_n = int(getattr(self, "crps_data_subsample", 0) or 0)
        if max_n > 0 and arr.size > max_n:
            base_seed = 0 if self.random_state is None else int(self.random_state)
            fam_seed = int(zlib.crc32(dist_name.encode("utf-8")) & 0xFFFFFFFF)
            rng = np.random.default_rng(int(base_seed) ^ fam_seed ^ 0xA5A5A5A5)
            idx = rng.choice(np.arange(arr.size), size=int(max_n), replace=False)
            arr = arr[idx]

        base_seed = 0 if self.random_state is None else int(self.random_state)
        fam_seed = int(zlib.crc32(dist_name.encode("utf-8")) & 0xFFFFFFFF)
        rng = np.random.default_rng(int(base_seed) ^ fam_seed ^ 0xC3C3C3C3)

        try:
            x = np.asarray(dist_obj.rvs(*params, size=int(m), random_state=rng), dtype=float).ravel()
            x2 = np.asarray(dist_obj.rvs(*params, size=int(m), random_state=rng), dtype=float).ravel()
        except Exception as exc:
            return None

        x = x[np.isfinite(x)]
        x2 = x2[np.isfinite(x2)]
        if x.size < 10 or x2.size < 10:
            return None

        # E|X - X'| term (independent sample estimate; O(m)).
        k = int(min(x.size, x2.size))
        term2 = float(np.mean(np.abs(x[:k] - x2[:k])))

        # E|X - x_obs| term (O(n*m); bounded by subsample + small m).
        diffs = np.abs(x[:k, None] - arr[None, :])
        term1 = float(np.mean(diffs))

        crps = term1 - 0.5 * term2
        return float(crps) if np.isfinite(crps) else None

    @staticmethod
    def _infer_interval_delta_from_data(data_flat: np.ndarray) -> Optional[float]:
        """Infer a plausible heaping/rounding delta for interval-likelihood mode."""
        arr = np.asarray(data_flat, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size < 30:
            return None

        n_unique = int(np.unique(arr).size)
        if n_unique <= 2:
            return 1.0

        # Integer-like heuristic.
        frac_int = float(np.mean(np.isclose(arr, np.round(arr), atol=1e-8)))
        if frac_int >= 0.98 and n_unique <= int(max(12, 0.10 * arr.size)):
            return 1.0

        best_delta = None
        best_score = 0.0
        for delta in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
            snapped = np.isclose(arr / delta, np.round(arr / delta), atol=0.02)
            score = float(np.mean(snapped))
            if score > best_score:
                best_score = score
                best_delta = float(delta)

        return best_delta if best_delta is not None and best_score >= 0.40 else None

    def _randomized_pit_from_intervals(
        self,
        data_flat: np.ndarray,
        *,
        dist_obj: sps.rv_continuous,
        params: Tuple,
        dist_name: str,
        delta: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Randomized PIT for interval/heaped observations.

        Returns:
            (u, interval_prob) where u ~ Uniform(F(lo), F(hi)) and interval_prob = F(hi)-F(lo).
        """
        arr = np.asarray(data_flat, dtype=float).ravel()
        half = 0.5 * float(delta)
        lo = arr - half
        hi = arr + half

        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            u_lo = np.asarray(dist_obj.cdf(lo, *params), dtype=float).ravel()
            u_hi = np.asarray(dist_obj.cdf(hi, *params), dtype=float).ravel()

        u_lo = np.nan_to_num(u_lo, nan=0.0, posinf=1.0, neginf=0.0)
        u_hi = np.nan_to_num(u_hi, nan=0.0, posinf=1.0, neginf=0.0)
        u_lo = np.clip(u_lo, 0.0, 1.0)
        u_hi = np.clip(u_hi, 0.0, 1.0)

        interval_prob = np.clip(u_hi - u_lo, 0.0, 1.0)

        base_seed = 0 if self.random_state is None else int(self.random_state)
        fam_seed = int(zlib.crc32(dist_name.encode("utf-8")) & 0xFFFFFFFF)
        rng = np.random.default_rng(int(base_seed) ^ fam_seed ^ 0x9E3779B9)

        u = u_lo + rng.random(size=interval_prob.size) * interval_prob
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        return u, interval_prob

    def _calculate_goodness_of_fit(
        self,
        data_original_nan_cleaned: np.ndarray,
        dist_obj: sps.rv_continuous,
        params_on_original_scale: Tuple,
        dist_name: str,
        *,
        compute_ad_p: bool = True,
    ) -> Dict[str, Any]:
        """Calculates various goodness-of-fit metrics using original data and original-scale parameters."""
        metrics = {}
        data_flat = data_original_nan_cleaned.flatten()
        
        min_samples_for_gof = 5 # Some GOF tests need a few samples
        if len(data_flat) < min_samples_for_gof: # Not enough data for GOF
            return {
                'ks_stat': 1.0,
                'ks_p': 0.0,
                'cvm_stat': float('inf'),
                'cvm_p': 0.0,
                'ad_stat': None,
                'ad_p': None,
                'qq_r2': None,
                'pp_r2': None,
                'pp_mae': None,
                'crps': None,
                'loglik': -float('inf'),
                'aic': float('inf'),
                'aicc': float('inf'),
                'bic': float('inf'),
            }
        try:
            interval_delta = None
            if bool(getattr(self, "interval_likelihood", False)):
                delta = float(getattr(self, "interval_delta", 0.0) or 0.0)
                if delta <= 0.0:
                    inferred = self._infer_interval_delta_from_data(data_flat)
                    if inferred is not None:
                        delta = float(inferred)
                if delta > 0.0:
                    interval_delta = float(delta)

            metrics['ad_stat'] = None
            metrics['ad_p'] = None
            metrics['qq_r2'] = None
            metrics['pp_r2'] = None
            metrics['pp_mae'] = None
            metrics['crps'] = None

            if interval_delta is not None:
                # Interval/heaped mode: randomized PIT uniformity tests + interval likelihood.
                u, interval_prob = self._randomized_pit_from_intervals(
                    data_flat,
                    dist_obj=dist_obj,
                    params=params_on_original_scale,
                    dist_name=dist_name,
                    delta=float(interval_delta),
                )

                with np.errstate(divide='ignore', invalid='ignore'):
                    metrics['ks_stat'], metrics['ks_p'] = kstest(u, "uniform")
                    cvm_res = cramervonmises(u, "uniform")
                    metrics['cvm_stat'], metrics['cvm_p'] = cvm_res.statistic, cvm_res.pvalue

                if self.compute_ad:
                    ad_stat = self._anderson_darling_stat_from_cdf(u)
                    metrics['ad_stat'] = float(ad_stat) if np.isfinite(ad_stat) else None
                    metrics['ad_p'] = None

                interval_prob = np.asarray(interval_prob, dtype=float).ravel()
                interval_prob = np.nan_to_num(interval_prob, nan=0.0, posinf=0.0, neginf=0.0)
                interval_prob = np.clip(interval_prob, 0.0, 1.0)
                loglik_sum = float(np.sum(np.log(np.maximum(interval_prob, 1e-300))))
                n_samples = int(len(data_flat))
                k_params = int(len(params_on_original_scale))

                metrics['loglik'] = loglik_sum
                metrics['aic'] = 2 * k_params - 2 * loglik_sum
                if (n_samples - k_params - 1) > 0:
                    metrics['aicc'] = metrics['aic'] + (2 * k_params * (k_params + 1)) / (n_samples - k_params - 1)
                else:
                    metrics['aicc'] = float('inf')
                metrics['bic'] = k_params * np.log(n_samples) - 2 * loglik_sum
            else:
                cdf_func = lambda x_val: dist_obj.cdf(x_val, *params_on_original_scale)
                pdf_func = lambda x_val: dist_obj.pdf(x_val, *params_on_original_scale)

                # KS and CvM tests (standard continuous GOF).
                with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings during GOF
                    metrics['ks_stat'], metrics['ks_p'] = kstest(data_flat, cdf_func)
                    cvm_res = cramervonmises(data_flat, cdf_func)
                    metrics['cvm_stat'], metrics['cvm_p'] = cvm_res.statistic, cvm_res.pvalue

                if self.compute_ad or self.compute_qq_pp:
                    xs = np.sort(np.asarray(data_flat, dtype=float))
                    u = np.asarray(dist_obj.cdf(xs, *params_on_original_scale), dtype=float)

                    if self.compute_ad:
                        ad_stat = self._anderson_darling_stat_from_cdf(u)
                        metrics['ad_stat'] = float(ad_stat) if np.isfinite(ad_stat) else None
                        if compute_ad_p and np.isfinite(ad_stat) and self.ad_bootstrap_samples > 0:
                            metrics['ad_p'] = self._bootstrap_ad_pvalue(
                                dist_obj=dist_obj,
                                params=params_on_original_scale,
                                n=int(xs.size),
                                ad_stat_obs=float(ad_stat),
                                dist_name=dist_name,
                            )

                    if self.compute_qq_pp:
                        qq_r2, pp_r2, pp_mae = self._qq_pp_diagnostics(
                            data_sorted=xs,
                            dist_obj=dist_obj,
                            params=params_on_original_scale,
                        )
                        metrics['qq_r2'] = qq_r2
                        metrics['pp_r2'] = pp_r2
                        metrics['pp_mae'] = pp_mae
                
                # Log-likelihood, AIC, BIC (point likelihood).
                log_probs = pdf_func(data_flat)
                log_probs = np.log(np.maximum(log_probs, 1e-150)) # Use very small floor for log
                valid_log_probs = log_probs[np.isfinite(log_probs)]
                
                if len(valid_log_probs) > 0:
                    loglik_sum = np.sum(valid_log_probs)
                    n_samples = len(data_flat) # Use N of original data for AIC/BIC
                    k_params = len(params_on_original_scale) 
                    
                    metrics['loglik'] = loglik_sum
                    metrics['aic'] = 2 * k_params - 2 * loglik_sum
                    if (n_samples - k_params - 1) > 0:
                        metrics['aicc'] = metrics['aic'] + (2 * k_params * (k_params + 1)) / (n_samples - k_params - 1)
                    else:
                        metrics['aicc'] = float('inf')
                    metrics['bic'] = k_params * np.log(n_samples) - 2 * loglik_sum
                else: # Fallback if log-likelihood calculation failed
                    metrics.update({'loglik': -float('inf'), 'aic': float('inf'), 'aicc': float('inf'), 'bic': float('inf')})

            if self.compute_crps:
                metrics['crps'] = self._compute_crps_mc(
                    data=data_flat,
                    dist_obj=dist_obj,
                    params=params_on_original_scale,
                    dist_name=dist_name,
                )
        except Exception as exc: # Broad catch for any GOF calculation errors
            metrics.update(
                {
                    'ks_stat': 1.0,
                    'ks_p': 0.0,
                    'cvm_stat': float('inf'),
                    'cvm_p': 0.0,
                    'ad_stat': None,
                    'ad_p': None,
                    'qq_r2': None,
                    'pp_r2': None,
                    'pp_mae': None,
                    'crps': None,
                    'loglik': -float('inf'),
                    'aic': float('inf'),
                    'aicc': float('inf'),
                    'bic': float('inf'),
                }
            )
        return metrics
    
    def select_best_distribution(self, data: np.ndarray, 
                               criterion: str = "simple", 
                               verbose: bool = False) -> Tuple[Optional[str], Optional[FitResult], List[FitResult]]:
        """
        Core method to fit all candidate distributions and select the best one.
        `criterion` can be 'simple', 'cvm_p', 'ks_p', 'bic', 'aic', 'aicc', 'cv',
        'cv_loglik', 'crps', or 'mnpo_oracle'.
        Returns the name of the best distribution, its FitResult object, and list of all results.
        """
        # Criterion-driven diagnostics: CRPS requires computing the score during fitting.
        crit = str(criterion).strip().lower()
        if crit == "crps" or (crit == "mnpo_oracle" and bool(getattr(self, "mnpo_include_crps", False))):
            self.compute_crps = True

        data_arr = np.asarray(data)
        data_clean_initial = data_arr[~np.isnan(data_arr)]
        
        if len(data_clean_initial) < 10:
            if verbose: print("Insufficient data (less than 10 non-NaN samples).")
            return None, None, []
        
        # Extract features from data *before* any distribution-specific robust processing
        features = DistributionFeatures.from_data(data_clean_initial)
        if verbose and features: self._print_verbose_features(features, len(data_clean_initial))

        dist_names_to_fit: List[str] = list(self.distributions.keys())
        if bool(getattr(self, "use_lmoment_prescreen", False)) and features is not None:
            dist_names_to_fit = self._lmoment_prescreen_distribution_names(dist_names_to_fit, features)

        all_fit_results: List[FitResult] = []
        
        # Choose between parallel and sequential fitting
        if self.n_jobs > 1 and len(dist_names_to_fit) > 1:
            # Parallel fitting
            if verbose: print(f"\nUsing parallel fitting with {self.n_jobs} workers...")
            all_fit_results = self._fit_distributions_parallel(data_clean_initial, features, dist_names=dist_names_to_fit, verbose=verbose)
        else:
            # Sequential fitting (original behavior)
            if verbose: print("\nUsing sequential fitting...")
            for dist_name_iter in dist_names_to_fit:
                if verbose: print(f"\nAttempting to fit: {dist_name_iter}")
                # Pass the initial NaN-cleaned data. _fit_single_distribution handles further prep.
                result_obj = self._fit_single_distribution(data_clean_initial.copy(), dist_name_iter, features)
                if verbose: self._print_verbose_fit_result(result_obj)
                all_fit_results.append(result_obj)
        
        # Perform Likelihood Ratio Tests if enabled
        lrt_results = []
        if self.use_lrt:
            lrt_results = self._perform_all_lrt_tests(data_clean_initial, all_fit_results, verbose)
        
        # Apply LRT results to adjust selection
        if lrt_results:
            all_fit_results = self._apply_lrt_adjustments(all_fit_results, lrt_results, verbose)
        
        # Filter for successful fits with valid scores
        successful_results = [r for r in all_fit_results if r.success and np.isfinite(r.simple_score)]
        
        if not successful_results:
            if verbose: print("\nNo distribution could be successfully fitted to the data.")
            return None, None, all_fit_results 

        if str(criterion).strip().lower() == "mnpo_oracle":
            if bool(getattr(self, "mnpo_include_preq", False)):
                self._compute_prequential_holdout_scores(
                    data_clean_initial,
                    successful_results,
                    random_state=getattr(self, "random_state", None),
                )
            self._apply_mnpo_oracle_weights(successful_results)
        
        # Sort results based on the chosen criterion
        sort_key, reverse_sort = self._get_sort_criteria(criterion)
        successful_results.sort(key=sort_key, reverse=reverse_sort)
        
        best_overall_result = successful_results[0]
        if verbose: self._print_verbose_best_result(best_overall_result, criterion, lrt_results)
        return best_overall_result.name, best_overall_result, all_fit_results
    
    def _fit_distributions_parallel(
        self,
        data: np.ndarray,
        features: Optional[DistributionFeatures],
        *,
        dist_names: Optional[Sequence[str]] = None,
        verbose: bool = False,
    ) -> List[FitResult]:
        """
        Fit all distributions in parallel using ProcessPoolExecutor.
        """
        names = list(dist_names) if dist_names is not None else list(self.distributions.keys())
        names = [n for n in names if n in self.distributions]

        # Prepare arguments for parallel processing
        features_dict = None
        if features:
            from dataclasses import asdict
            features_dict = asdict(features)
        
        # Configuration for worker processes
        config_dict = {
            'robust_mode': self.robust_mode,
            'use_adaptive_strategy': self.use_adaptive_strategy,
            'use_cv': self.use_cv,
            'fit_estimator': getattr(self, "fit_estimator", "mle"),
            'mps_maxiter': getattr(self, "mps_maxiter", 250),
            'mps_tol': getattr(self, "mps_tol", 1e-6),
            'compute_ad': self.compute_ad,
            'ad_bootstrap_samples': self.ad_bootstrap_samples,
            'compute_qq_pp': self.compute_qq_pp,
            'interval_likelihood': bool(getattr(self, "interval_likelihood", False)),
            'interval_delta': float(getattr(self, "interval_delta", 0.0) or 0.0),
            'compute_crps': getattr(self, "compute_crps", False),
            'crps_mc_samples': getattr(self, "crps_mc_samples", 96),
            'crps_data_subsample': getattr(self, "crps_data_subsample", 256),
            'random_state': self.random_state,
        }
        
        # Create arguments list for parallel processing.
        # Derive per-distribution seeds for statistical independence (VAL12_Suggestions §1.2).
        _base_seed = int(config_dict.get('random_state', 0) or 0)
        args_list = []
        for _dist_idx, dist_name in enumerate(names):
            _per_dist_config = dict(config_dict)
            _per_dist_config['random_state'] = int((_base_seed + _dist_idx) % (2**31 - 1))
            args_list.append((dist_name, data.copy(), features_dict, _per_dist_config))
        
        all_fit_results = []
        
        try:
            # Use ProcessPoolExecutor for parallel fitting
            max_workers = int(max(1, min(int(self.n_jobs), len(args_list))))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs
                future_to_dist = {
                    executor.submit(self._fit_distribution_worker, args): args[0] 
                    for args in args_list
                }

                pending_futures = set(future_to_dist.keys())
                try:
                    # Global guard keeps orchestration from stalling indefinitely.
                    for future in as_completed(
                        pending_futures,
                        timeout=300.0 * max(1, len(pending_futures)),
                    ):
                        pending_futures.discard(future)
                        dist_name = future_to_dist[future]
                        try:
                            result = future.result(timeout=300)
                            all_fit_results.append(result)
                            if verbose:
                                self._print_verbose_fit_result(result)
                        except FuturesTimeoutError:
                            if verbose:
                                print(f"  {dist_name} timed out after 300s")
                            future.cancel()
                            all_fit_results.append(
                                FitResult(
                                    name=dist_name,
                                    params=None,
                                    transform_info=TransformInfo(),
                                    success=False,
                                    error="Parallel execution timeout after 300s",
                                )
                            )
                        except Exception as exc:
                            if verbose:
                                print(f"  {dist_name} generated an exception: {exc}")
                            all_fit_results.append(
                                FitResult(
                                    name=dist_name,
                                    params=None,
                                    transform_info=TransformInfo(),
                                    success=False,
                                    error=f"Parallel execution error: {str(exc)}",
                                )
                            )
                except FuturesTimeoutError:
                    # as_completed global timeout: mark all unfinished jobs as timed out.
                    for future in list(pending_futures):
                        dist_name = future_to_dist[future]
                        future.cancel()
                        if verbose:
                            print(f"  {dist_name} timed out after 300s (global wait exceeded)")
                        all_fit_results.append(
                            FitResult(
                                name=dist_name,
                                params=None,
                                transform_info=TransformInfo(),
                                success=False,
                                error="Parallel execution timeout after 300s",
                            )
                        )
        
        except Exception as e:
            if verbose:
                print(f"Parallel processing failed: {e}. Falling back to sequential processing...")
            # Fallback to sequential processing
            all_fit_results = []
            for dist_name in names:
                if verbose: print(f"\nAttempting to fit: {dist_name}")
                result_obj = self._fit_single_distribution(data.copy(), dist_name, features)
                if verbose: self._print_verbose_fit_result(result_obj)
                all_fit_results.append(result_obj)
        
        # Sort results by distribution name to ensure consistent ordering
        all_fit_results.sort(key=lambda x: x.name)
        
        return all_fit_results

    def _perform_all_lrt_tests(self, data: np.ndarray, fit_results: List[FitResult], 
                              verbose: bool = False) -> List[LRTResult]:
        """
        Perform all applicable LRT tests between nested distributions.
        """
        lrt_results = []
        
        # Create a mapping from distribution name to fit result for easy lookup
        result_map = {r.name: r for r in fit_results if r.success and r.params is not None}
        
        for complex_dist, nested_list in self.nested_models.items():
            if complex_dist not in result_map:
                continue
                
            complex_result = result_map[complex_dist]
            
            for simple_dist, constraint_desc in nested_list:
                if simple_dist not in result_map:
                    continue
                    
                simple_result = result_map[simple_dist]
                
                lrt_result = self._perform_lrt(
                    data, simple_dist, complex_dist,
                    simple_result.params, complex_result.params
                )
                
                if lrt_result is not None:
                    lrt_results.append(lrt_result)
                    if verbose:
                        self._print_verbose_lrt_result(lrt_result)
        
        return lrt_results

    def _apply_lrt_adjustments(self, fit_results: List[FitResult], lrt_results: List[LRTResult],
                              verbose: bool = False) -> List[FitResult]:
        """
        Apply LRT results to adjust distribution selection by penalizing complex models
        when the simpler model is not significantly worse.
        """
        # Create a penalty map for distributions that should be penalized
        penalty_map = {}
        
        for lrt in lrt_results:
            if not lrt.prefer_complex:
                # If LRT doesn't prefer the complex model, penalize it
                penalty_map[lrt.complex_dist] = penalty_map.get(lrt.complex_dist, 0) + 0.1
                if verbose:
                    print(f"LRT penalty applied to {lrt.complex_dist}: +{0.1} (total: {penalty_map[lrt.complex_dist]})")
        
        # Apply penalties to fit results
        adjusted_results = []
        for result in fit_results:
            if result.name in penalty_map and result.success:
                # Create a copy of the result with adjusted simple_score
                # Note: We can't modify the property directly, so we adjust the components
                penalty = penalty_map[result.name]
                
                # Create a new FitResult with penalty applied via reduced feature_bonus
                adjusted_result = FitResult(
                    name=result.name,
                    params=result.params,
                    transform_info=result.transform_info,
                    ks_stat=result.ks_stat,
                    ks_p=result.ks_p,
                    cvm_stat=result.cvm_stat,
                    cvm_p=result.cvm_p,
                    ad_stat=result.ad_stat,
                    ad_p=result.ad_p,
                    qq_r2=result.qq_r2,
                    pp_r2=result.pp_r2,
                    pp_mae=result.pp_mae,
                    crps=result.crps,
                    aic=result.aic,
                    aicc=result.aicc,
                    bic=result.bic,
                    loglik=result.loglik,
                    cv_result=result.cv_result,
                    preq_loglik_mean=getattr(result, "preq_loglik_mean", None),
                    success=result.success,
                    error=result.error,
                    feature_bonus=max(0.0, result.feature_bonus - penalty),  # Apply penalty
                    fit_method=result.fit_method,
                    mnpo_weight=result.mnpo_weight,
                )
                adjusted_results.append(adjusted_result)
            else:
                adjusted_results.append(result)
        
        return adjusted_results

    @staticmethod
    def _normalize_vector_01(values: Sequence[float]) -> np.ndarray:
        """Min-max normalize to [0,1] with safe fallback."""
        return _mnpo_normalize_vector_01(values)

    def _pairwise_pref_from_scalar(
        self,
        scalar_i: float,
        scalar_j: float,
        *,
        temperature: Optional[float] = None,
    ) -> float:
        """Preference probability from scalar oracle values."""
        return float(
            _mnpo_pairwise_pref_from_scalar(
                float(scalar_i),
                float(scalar_j),
                tie_margin=float(self.mnpo_pairwise_tie_margin),
                temperature=temperature,
            )
        )

    def _fit_tritrust_weights_df(self, oracle_matrices: Dict[str, np.ndarray], reference: str) -> Dict[str, float]:
        """TriTrust-style trust/ignore calibration from agreement with a reference oracle."""
        return dict(
            _mnpo_fit_tritrust_weights(
                oracle_matrices,
                reference=str(reference),
                allow_negative=False,
                no_flip_oracles=None,
                ref_delta_threshold=0.05,
                oracle_delta_threshold=0.03,
                reliability_threshold=0.10,
            )
        )

    @staticmethod
    def _aggregate_payoff_matrix_df(oracle_matrices: Dict[str, np.ndarray], oracle_weights: Dict[str, float]) -> np.ndarray:
        """Aggregate oracle preferences into an anti-symmetric payoff matrix."""
        return _mnpo_aggregate_payoff_matrix(oracle_matrices, oracle_weights)

    def _mirror_descent_mnpo_df(self, payoff: np.ndarray, reference_prior: np.ndarray) -> np.ndarray:
        """Reference-regularized mirror descent on the candidate simplex."""
        return np.asarray(
            _mnpo_mirror_descent_reference_regularized(
                np.asarray(payoff, dtype=float),
                np.asarray(reference_prior, dtype=float),
                steps=int(self.mnpo_mirror_descent_steps),
                eta=float(self.mnpo_mirror_descent_eta),
                lambda_=float(self.mnpo_mirror_descent_lambda),
                tol_kl=1e-7,
                return_history=False,
            ),
            dtype=float,
        )

    def _apply_mnpo_oracle_weights(self, results: List[FitResult]) -> None:
        """Compute MNPO equilibrium weights for the given successful FitResults."""
        m = int(len(results))
        if m == 0:
            return
        if m == 1:
            results[0].mnpo_weight = 1.0
            return

        # 1) Build scalar oracle scores.
        gof_raw: List[float] = []
        for r in results:
            pvals: List[float] = []
            if np.isfinite(r.cvm_p):
                pvals.append(float(r.cvm_p))
            if np.isfinite(r.ks_p):
                pvals.append(float(r.ks_p))
            if self.compute_ad and r.ad_p is not None and np.isfinite(r.ad_p):
                pvals.append(float(r.ad_p))
            gof_raw.append(float(min(pvals)) if pvals else 0.0)
        gof = self._normalize_vector_01(gof_raw)

        pred_raw: List[float] = []
        has_pred = False
        for r in results:
            v = float("nan")
            if r.cv_result is not None and int(getattr(r.cv_result, "successful_folds", 0)) > 0:
                v = float(getattr(r.cv_result, "cv_loglik_mean", float("nan")))
            if np.isfinite(v):
                has_pred = True
            pred_raw.append(v)
        pred = self._normalize_vector_01(pred_raw) if has_pred else np.full(m, 0.5, dtype=float)

        preq_raw: List[float] = []
        has_preq = False
        if bool(getattr(self, "mnpo_include_preq", False)):
            for r in results:
                v = float("nan")
                if getattr(r, "preq_loglik_mean", None) is not None:
                    try:
                        v = float(getattr(r, "preq_loglik_mean"))
                    except Exception as exc:
                        v = float("nan")
                if np.isfinite(v):
                    has_preq = True
                preq_raw.append(v)
        preq = self._normalize_vector_01(preq_raw) if has_preq else np.full(m, 0.5, dtype=float)

        par_raw: List[float] = []
        has_par = False
        for r in results:
            bic = float(getattr(r, "bic", float("inf")))
            v = -bic if np.isfinite(bic) else float("nan")
            if np.isfinite(v):
                has_par = True
            par_raw.append(v)
        par = self._normalize_vector_01(par_raw) if has_par else np.full(m, 0.5, dtype=float)

        crps_raw: List[float] = []
        has_crps = False
        if bool(getattr(self, "mnpo_include_crps", False)) and bool(getattr(self, "compute_crps", False)):
            for r in results:
                v = float("nan")
                if r.crps is not None and np.isfinite(float(r.crps)):
                    v = -float(r.crps)  # lower CRPS is better
                if np.isfinite(v):
                    has_crps = True
                crps_raw.append(v)
        crps = self._normalize_vector_01(crps_raw) if has_crps else np.full(m, 0.5, dtype=float)

        # 2) Pairwise preference matrices (optionally QRE-smoothed).
        oracle_matrices: Dict[str, np.ndarray] = {}
        oracle_pairwise_meta: Dict[str, Any] = {}
        oracle_score_vectors: Dict[str, np.ndarray] = {"gof": gof}
        if has_pred:
            oracle_score_vectors["predictive"] = pred
        if has_preq:
            oracle_score_vectors["prequential"] = preq
        if has_par:
            oracle_score_vectors["parsimony"] = par
        if has_crps:
            oracle_score_vectors["crps"] = crps

        for name, scores in oracle_score_vectors.items():
            mat, meta = _mnpo_matrix_from_scalar_scores(
                np.asarray(scores, dtype=float),
                tie_margin=float(self.mnpo_pairwise_tie_margin),
                use_qre_smoothing=bool(getattr(self, "mnpo_use_qre_smoothing", False)),
                qre_temperature_gamma=float(getattr(self, "mnpo_qre_temperature_gamma", 1.0) or 1.0),
            )
            oracle_matrices[str(name)] = np.asarray(mat, dtype=float)
            oracle_pairwise_meta[str(name)] = dict(meta)

        # 3) TriTrust weights (optional) relative to predictive (preferred) else GOF.
        if "predictive" in oracle_matrices:
            reference = "predictive"
        elif "prequential" in oracle_matrices:
            reference = "prequential"
        else:
            reference = "gof"
        if self.mnpo_use_tritrust:
            oracle_weights = self._fit_tritrust_weights_df(oracle_matrices, reference=reference)
        else:
            oracle_weights = {name: 1.0 for name in oracle_matrices}

        oracle_weights_tritrust = dict(oracle_weights)
        oracle_redundancy_meta = None
        if bool(getattr(self, "mnpo_use_oracle_redundancy_penalty", False)):
            oracle_weights, oracle_redundancy_meta = _mnpo_apply_oracle_redundancy_penalty(
                dict(oracle_weights),
                oracle_score_vectors,
            )

        payoff = self._aggregate_payoff_matrix_df(oracle_matrices, oracle_weights)
        prior = np.full(m, 1.0 / float(m), dtype=float)
        p_star = self._mirror_descent_mnpo_df(payoff, prior)

        tremble_meta = None
        if bool(getattr(self, "mnpo_compute_tremble_sensitivity", False)) and m > 1:
            eps = 0.01
            trembled = _mnpo_tremble_oracle_matrices(oracle_matrices, epsilon=float(eps))
            payoff_eps = self._aggregate_payoff_matrix_df(trembled, oracle_weights)
            p_eps = self._mirror_descent_mnpo_df(payoff_eps, prior)
            diff = np.asarray(p_eps, dtype=float).ravel() - np.asarray(p_star, dtype=float).ravel()
            abs_shift = np.abs(diff)
            tremble_meta = {
                "epsilon": float(eps),
                "l1": float(np.sum(abs_shift)),
                "l2": float(np.sqrt(np.sum(diff * diff))),
                "linf": float(np.max(abs_shift)) if abs_shift.size else 0.0,
            }

        # Emit last-run diagnostics for attribution/debugging (JSON-safe).
        try:
            self.mnpo_diagnostics_ = {
                "reference_oracle": str(reference),
                "oracle_scores": {k: [float(v) for v in np.asarray(s).ravel()] for k, s in oracle_score_vectors.items()},
                "oracle_pairwise_meta": dict(oracle_pairwise_meta),
                "oracle_weights_tritrust": {k: float(v) for k, v in oracle_weights_tritrust.items()},
                "oracle_weights": {k: float(v) for k, v in oracle_weights.items()},
                "oracle_redundancy_meta": dict(oracle_redundancy_meta) if isinstance(oracle_redundancy_meta, dict) else {},
                "tremble_sensitivity": dict(tremble_meta) if isinstance(tremble_meta, dict) else {},
            }
        except Exception as exc:
            self.mnpo_diagnostics_ = {}

        for idx, r in enumerate(results):
            r.mnpo_weight = float(p_star[idx])

    def _get_sort_criteria(self, criterion_str: str):
        """Helper to get lambda sort key and reverse flag for sorting FitResults."""
        criterion_str = criterion_str.lower()
        if criterion_str == "cvm_p": return (lambda r: r.cvm_p, True) # Higher p-value is better
        if criterion_str == "ks_p": return (lambda r: r.ks_p, True)   # Higher p-value is better
        if criterion_str == "bic": return (lambda r: r.bic, False)    # Lower BIC is better
        if criterion_str == "aic": return (lambda r: r.aic, False)    # Lower AIC is better
        if criterion_str == "aicc": return (lambda r: r.aicc, False)  # Lower AICc is better
        if criterion_str == "cv": return (lambda r: r.cv_result.cv_score if r.cv_result else -1, True)  # Higher CV score is better
        if criterion_str == "cv_loglik": return (lambda r: r.cv_result.cv_loglik_mean if r.cv_result else -float('inf'), True)  # Higher CV log-likelihood is better
        if criterion_str == "crps": return (lambda r: r.crps if (r.crps is not None and np.isfinite(r.crps)) else float("inf"), False)
        if criterion_str == "mnpo_oracle": return (lambda r: r.mnpo_weight if r.mnpo_weight is not None else -1.0, True)
        return (lambda r: r.simple_score, False) # Default: simple_score (lower is better)

    def _print_verbose_features(self, features: DistributionFeatures, n_samples: int):
        """Prints extracted data features if verbose mode is on."""
        print(f"Data characteristics (N={n_samples}):")
        print(f"  Mean={features.mean:.3g}, Std={features.std:.3g}, CV={features.cv:.3g}")
        print(f"  Skew={features.skewness:.3g}, Kurtosis(excess)={features.excess_kurtosis:.3g}")
        if features.lognormal_score > 0: print(f"  Lognormal Score(p)={features.lognormal_score:.3g}")
        # Add other feature scores if desired

    def _print_verbose_fit_result(self, result: FitResult):
        """Prints individual fit result details if verbose mode is on."""
        if result.success:
            cv_info = ""
            if result.cv_result is not None:
                cv_info = f", CV_score={result.cv_result.cv_score:.3f}({result.cv_result.successful_folds}/{result.cv_result.total_folds})"
            print(f"  Fit: {result.name}, SimpleScore={result.simple_score:.3f}, CvM_p={result.cvm_p:.3f}, Bonus={result.feature_bonus:.2f}{cv_info}")
            # print(f"     Params: {result.params}") # Can be very long
        else:
            print(f"  Fit: {result.name} FAILED. Error: {result.error}")

    def _print_verbose_best_result(self, best_result: FitResult, criterion: str, lrt_results: List[LRTResult]):
        """Prints details of the best selected distribution if verbose mode is on."""
        print(f"\n--- Best Distribution Selection (Criterion: {criterion}) ---")
        print(f"Best Fit: {best_result.name}")
        print(f"  Simple Score: {best_result.simple_score:.4f}")
        print(f"  Parameters (original scale): {best_result.params}")
        print(f"  CvM (p={best_result.cvm_p:.4f}, stat={best_result.cvm_stat:.3g})")
        print(f"  KS  (p={best_result.ks_p:.4f}, stat={best_result.ks_stat:.3g})")
        print(f"  BIC: {best_result.bic:.3g}, AIC: {best_result.aic:.3g}, LogLik: {best_result.loglik:.3g}")
        print(f"  Feature Bonus: {best_result.feature_bonus:.3f}")
        if best_result.cv_result is not None:
            print(f"  CV Results: Mean LogLik={best_result.cv_result.cv_loglik_mean:.3g}, CV Score={best_result.cv_result.cv_score:.3f}")
            print(f"  CV Folds: {best_result.cv_result.successful_folds}/{best_result.cv_result.total_folds} successful")
        if best_result.transform_info.shifted or best_result.transform_info.scaled:
            print(f"  Transformations for fitting: Shifted={best_result.transform_info.shifted}, Scaled={best_result.transform_info.scaled}")
        if lrt_results:
            print("\n--- LRT Results ---")
            for lrt in lrt_results:
                print(f"LRT between {lrt.simple_dist} and {lrt.complex_dist}:")
                print(f"  LRT Statistic: {lrt.lrt_statistic:.3g}")
                print(f"  p-value: {lrt.p_value:.4f}")
                print(f"  Prefer Complex: {lrt.prefer_complex}")
        print("----------------------------------------------------")

    def _perform_lrt(self, data: np.ndarray, simple_dist: str, complex_dist: str,
                     simple_params: Tuple, complex_params: Tuple) -> Optional[LRTResult]:
        """
        Perform Likelihood Ratio Test between nested distributions.
        """
        try:
            simple_obj = self.distributions[simple_dist]
            complex_obj = self.distributions[complex_dist]
            
            # Calculate log-likelihoods
            simple_loglik = np.sum(simple_obj.logpdf(data, *simple_params))
            complex_loglik = np.sum(complex_obj.logpdf(data, *complex_params))
            
            # Ensure both log-likelihoods are finite
            if not (np.isfinite(simple_loglik) and np.isfinite(complex_loglik)):
                return None
            
            # Calculate LRT statistic: -2 * (L_simple - L_complex)
            lrt_stat = -2 * (simple_loglik - complex_loglik)
            
            # Degrees of freedom = difference in number of parameters
            df = len(complex_params) - len(simple_params)
            
            if df <= 0:
                return None  # Complex model should have more parameters
            
            # Calculate p-value using chi-squared distribution
            p_value = 1 - chi2.cdf(lrt_stat, df)
            
            # Prefer complex model if p-value < 0.05 (conventional significance level)
            prefer_complex = p_value < 0.05
            
            return LRTResult(
                simple_dist=simple_dist,
                complex_dist=complex_dist,
                lrt_statistic=lrt_stat,
                p_value=p_value,
                df=df,
                prefer_complex=prefer_complex,
                simple_loglik=simple_loglik,
                complex_loglik=complex_loglik
            )
        except Exception as exc:
            return None

    def _perform_cross_validation(self, data: np.ndarray, dist_name: str,
                                features: Optional[DistributionFeatures]) -> Optional[CVResult]:
        """
        Perform leave-one-out cross-validation for distribution fitting.
        """
        try:
            if len(data) < 10:  # Need sufficient data for meaningful CV
                return None
            
            n = len(data)
            cv_logliks = []
            successful_folds = 0
            
            # Limit LOOCV to a maximum number of folds for performance reasons if data is very large
            # For example, max 200 folds. If n > 200, sample 200 indices for LOOCV.
            max_cv_folds = 200 
            indices_to_use = range(n)
            if n > max_cv_folds:
                # FIX CRITICAL-001: Use seeded RNG for reproducibility (T-AUDIT-001-FIX-001)
                base_seed = 0 if self.random_state is None else int(self.random_state)
                rng = np.random.default_rng(int(base_seed) ^ 0xA0A0A001)  # Unique seed for LOOCV sampling
                indices_to_use = rng.choice(n, size=max_cv_folds, replace=False)

            for i in indices_to_use:
                # Create training set (all data except point i)
                train_data = np.concatenate([data[:i], data[i+1:]])
                test_point = data[i]
                
                try:
                    # Fit distribution on training data, ensuring CV is not re-triggered
                    train_result = self._fit_single_distribution(
                        train_data, dist_name, features, is_cv_fold=True, compute_gof=False
                    )
                    
                    if train_result.success and train_result.params is not None:
                        # Calculate log-likelihood of test point
                        dist_obj = self.distributions[dist_name]
                        test_loglik = dist_obj.logpdf(test_point, *train_result.params)
                        
                        if np.isfinite(test_loglik):
                            cv_logliks.append(test_loglik)
                            successful_folds += 1
                except Exception as exc:
                    continue  # Skip this fold if fitting fails
            
            if len(cv_logliks) == 0:
                return None
            
            cv_logliks_arr = np.asarray(cv_logliks, dtype=float).ravel()
            cv_logliks_arr = cv_logliks_arr[np.isfinite(cv_logliks_arr)]
            if cv_logliks_arr.size == 0:
                return None
            cv_mean = float(np.mean(cv_logliks_arr))
            cv_std = float(np.std(cv_logliks_arr, ddof=1)) if cv_logliks_arr.size > 1 else 0.0
            
            # Create composite CV score (higher is better)
            # Normalize by scaling mean log-likelihood to [0, 1] range approximately
            # Use sigmoid-like transformation to handle varying ranges
            cv_score = 1.0 / (1.0 + np.exp(-cv_mean)) if np.isfinite(cv_mean) else 0.0
            
            return CVResult(
                dist_name=dist_name,
                cv_loglik_mean=cv_mean,
                cv_loglik_std=cv_std,
                cv_score=cv_score,
                successful_folds=successful_folds,
                total_folds=len(indices_to_use),  # Use actual number of folds performed
                cv_logliks=[float(v) for v in cv_logliks_arr.tolist()],
            )
        except Exception as exc:
            return None

    def _compute_prequential_holdout_scores(
        self,
        data: np.ndarray,
        results: List[FitResult],
        *,
        random_state: Optional[int] = None,
    ) -> None:
        """Compute a cheap prequential/holdout predictive log-likelihood (opt-in).

        This is intended as a lightweight predictive oracle for MNPO when full
        LOOCV is disabled or too expensive. It fits each candidate family on a
        deterministic train split and evaluates mean log-likelihood on a
        deterministic holdout set.
        """
        if not results:
            return

        arr = np.asarray(data, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        n = int(arr.size)
        if n < max(10, int(getattr(self, "preq_min_train", 20)) + 1):
            return

        seed = 0 if random_state is None else int(random_state)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)

        holdout_fraction = float(getattr(self, "preq_holdout_fraction", 0.20))
        max_test = int(getattr(self, "preq_max_test_points", 128) or 128)
        min_train = int(getattr(self, "preq_min_train", 20) or 20)

        n_test = int(max(1, round(holdout_fraction * n)))
        n_test = int(min(n_test, max_test))
        n_train = n - n_test
        if n_train < min_train:
            n_test = max(1, n - min_train)
            n_train = n - n_test
        if n_train < 3 or n_test < 1:
            return

        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        train_data = arr[train_idx]
        test_data = arr[test_idx]

        train_features = DistributionFeatures.from_data(train_data)

        for r in results:
            if not bool(getattr(r, "success", False)):
                continue
            dist_name = str(getattr(r, "name", "") or "")
            if dist_name not in self.distributions:
                continue
            try:
                fit_res = self._fit_single_distribution(
                    train_data,
                    dist_name,
                    train_features,
                    is_cv_fold=True,
                    compute_gof=False,
                )
                if not bool(getattr(fit_res, "success", False)) or getattr(fit_res, "params", None) is None:
                    continue
                dist_obj = self.distributions[dist_name]
                ll = dist_obj.logpdf(test_data, *fit_res.params)
                ll = np.asarray(ll, dtype=float).ravel()
                ll = ll[np.isfinite(ll)]
                if ll.size == 0:
                    continue
                r.preq_loglik_mean = float(np.mean(ll))
            except Exception as exc:
                continue

    def _print_verbose_lrt_result(self, lrt_result: LRTResult):
        """Prints details of an individual LRT result if verbose mode is on."""
        print(f"LRT between {lrt_result.simple_dist} and {lrt_result.complex_dist}:")
        print(f"  LRT Statistic: {lrt_result.lrt_statistic:.3g}")
        print(f"  p-value: {lrt_result.p_value:.4f}")
        print(f"  Prefer Complex: {lrt_result.prefer_complex}")
        print("----------------------------------------------------")

    def _get_initial_parameter_guess(self, data: np.ndarray, dist_name: str, 
                                   features: Optional[DistributionFeatures]) -> Optional[Tuple]:
        """
        Generate initial parameter guesses for MLE fitting based on method of moments
        or distribution features to improve convergence.
        """
        if len(data) < 3 or features is None:
            return None
            
        try:
            mean_val = features.mean
            std_val = features.std
            skew_val = features.skewness
            
            if dist_name == "norm":
                return (mean_val, std_val)
            
            elif dist_name == "expon":
                # For exponential: rate = 1/mean, but scipy uses (loc, scale) where scale = mean
                loc_guess = np.min(data) if np.min(data) >= 0 else 0
                scale_guess = mean_val - loc_guess if mean_val > loc_guess else 1.0
                return (loc_guess, max(scale_guess, 0.1))
            
            elif dist_name == "gamma":
                # Method of moments for gamma: shape = mean^2/var, scale = var/mean
                if std_val > 0:
                    var_val = std_val**2
                    shape_guess = (mean_val**2) / var_val
                    scale_guess = var_val / mean_val
                    loc_guess = 0.0
                    return (max(shape_guess, 0.1), loc_guess, max(scale_guess, 0.1))
                return (1.0, 0.0, 1.0)
            
            elif dist_name == "weibull_min":
                # Rough approximation: shape ≈ 1.2 / CV for CV < 1
                if features.cv > 0 and features.cv < 2:
                    shape_guess = max(0.5, min(5.0, 1.2 / features.cv))
                else:
                    shape_guess = 1.0
                loc_guess = np.min(data) * 0.9 if np.min(data) > 0 else 0
                scale_guess = mean_val - loc_guess if mean_val > loc_guess else 1.0
                return (shape_guess, loc_guess, max(scale_guess, 0.1))
            
            elif dist_name == "lognorm":
                # For lognormal, log(data) should be normal
                positive_data = data[data > 0]
                if len(positive_data) > 0:
                    log_data = np.log(positive_data)
                    s_guess = np.std(log_data)  # Shape parameter
                    loc_guess = 0.0
                    scale_guess = np.exp(np.mean(log_data))
                    return (max(s_guess, 0.1), loc_guess, max(scale_guess, 0.1))
                return (1.0, 0.0, 1.0)
            
            elif dist_name == "t":
                # For t-distribution, estimate df from excess kurtosis
                if features.excess_kurtosis > 0:
                    # Excess kurtosis = 6/(df-4) for df > 4
                    df_guess = 4 + 6 / max(features.excess_kurtosis, 0.1)
                    df_guess = max(1.1, min(100, df_guess))
                else:
                    df_guess = 10.0
                return (df_guess, mean_val, std_val)
            
            elif dist_name == "laplace":
                # Laplace: scale = std/sqrt(2)
                return (mean_val, std_val / np.sqrt(2))
            
            elif dist_name == "beta":
                # Beta distribution method of moments
                if 0 < mean_val < 1 and std_val > 0:
                    var_val = std_val**2
                    # Ensure the variance constraint for beta distribution
                    max_var = mean_val * (1 - mean_val)
                    if var_val >= max_var:
                        var_val = max_var * 0.9
                    
                    common_term = (mean_val * (1 - mean_val) / var_val) - 1
                    if common_term > 0:
                        a_guess = mean_val * common_term
                        b_guess = (1 - mean_val) * common_term
                        return (max(a_guess, 0.1), max(b_guess, 0.1), 0.0, 1.0)
                return (1.0, 1.0, 0.0, 1.0)
            
            elif dist_name == "pareto":
                # Pareto: shape parameter from mean and std
                if mean_val > 0 and std_val > 0:
                    cv_sq = (std_val / mean_val)**2
                    if cv_sq > 1:  # Required for Pareto
                        shape_guess = 1 + np.sqrt(1 + 1/cv_sq)
                        shape_guess = max(1.1, min(10, shape_guess))
                        return (shape_guess, 0.0, 1.0)
                return (1.5, 0.0, 1.0)
            
            # For other distributions, return None (use scipy defaults)
            return None
            
        except Exception as exc:
            return None

    def _get_fit_bounds_and_constraints(self, data: np.ndarray, dist_name: str,
                                      features: Optional[DistributionFeatures]) -> Dict[str, Any]:
        """
        Generate parameter bounds and constraints for fitting based on distribution
        characteristics and data features.
        """
        fit_kwargs = {}
        
        try:
            if dist_name == "expon":
                # Exponential should start at or near minimum value
                if features and features.is_positive:
                    min_val = np.min(data)
                    if min_val > 0:
                        fit_kwargs['floc'] = min_val * 0.99  # Slightly below minimum
                    else:
                        fit_kwargs['floc'] = 0
                
            elif dist_name == "beta":
                # Beta should be on [0,1] interval - already handled in _prepare_data
                # If data was scaled to [0,1], fix loc and scale
                if np.min(data) >= -1e-9 and np.max(data) <= 1.0 + 1e-9:
                    fit_kwargs['floc'] = 0
                    fit_kwargs['fscale'] = 1
                    
            elif dist_name == "uniform":
                # For uniform, we can fix loc and scale based on min/max
                if features and abs(features.skewness) < 0.1:  # Very symmetric
                    min_val, max_val = np.min(data), np.max(data)
                    data_range = max_val - min_val
                    if data_range > 1e-6:
                        fit_kwargs['floc'] = min_val
                        fit_kwargs['fscale'] = data_range
            
            elif dist_name in ["gamma", "weibull_min", "lognorm", "pareto"]:
                # These distributions need positive scale
                if features and features.is_positive:
                    # We can sometimes fix location to 0 for positive data
                    min_val = np.min(data)
                    if min_val > 1e-6:  # Data is clearly positive
                        # Don't fix location, but we could add bounds if scipy supported it
                        pass
            
            elif dist_name == "powerlaw":
                # Powerlaw needs data in [0,1] or needs scaling
                if np.min(data) >= 0 and np.max(data) <= 1.001:
                    fit_kwargs['floc'] = 0
                    fit_kwargs['fscale'] = 1
                    
            elif dist_name == "triang":
                # Triangle distribution: loc ≤ data ≤ loc + scale
                min_val, max_val = np.min(data), np.max(data)
                data_range = max_val - min_val
                if data_range > 1e-6:
                    fit_kwargs['floc'] = min_val
                    fit_kwargs['fscale'] = data_range
                    
        except Exception as exc:
            pass  # Return empty dict if any error occurs
            
        return fit_kwargs

    # Compatibility method for the FeatureEngineeringPipeline
    def fit_and_select_distribution(self, data: np.ndarray, 
                                  criterion: str = "cvm") -> Tuple[Optional[str], Optional[Tuple], Optional[float], Any]:
        """
        Compatibility method for FeatureEngineeringPipeline.
        Maps pipeline criteria to internal selection criteria and returns results
        in the expected format: (best_dist_name, best_params_tuple, score_value, extra_info_value).
        """
        criterion_map = {"cvm": "simple", "ks": "simple", "bic": "bic", "aic": "aic"}
        mapped_criterion = criterion_map.get(criterion.lower(), "simple")
        
        try:
            best_name, best_result_obj, _ = self.select_best_distribution(
                data, criterion=mapped_criterion, verbose=False # Verbose off for pipeline use
            )
            
            if not best_name or not best_result_obj: return None, None, None, None

            score_val: Optional[float] = None
            extra_val: Any = None

            if criterion.lower() == "cvm":
                score_val, extra_val = best_result_obj.cvm_p, best_result_obj.cvm_stat
            elif criterion.lower() == "ks":
                score_val, extra_val = best_result_obj.ks_p, best_result_obj.ks_stat
            elif criterion.lower() == "bic": score_val = best_result_obj.bic
            elif criterion.lower() == "aic": score_val = best_result_obj.aic
            else: # Default for "simple" or unmapped: return simple_score and key metrics
                score_val = best_result_obj.simple_score 
                extra_val = {"cvm_p": best_result_obj.cvm_p, "ks_p": best_result_obj.ks_p}
            
            return best_name, best_result_obj.params, score_val, extra_val
        except Exception as exc: # Broad catch for unforeseen errors
            return None, None, None, None

    def _calculate_profile_likelihood_ci(self, data: np.ndarray, dist_obj: sps.rv_continuous,
                                        fitted_params: Tuple, param_index: int = 0,
                                        confidence_level: float = 0.95) -> Optional[Tuple[float, float]]:
        """
        Calculate confidence interval for a specific parameter using profile likelihood.
        This gives insights into parameter uncertainty and model stability.
        """
        try:
            if len(data) < 10:  # Need sufficient data
                return None
            
            # Calculate the maximum log-likelihood
            max_loglik = np.sum(dist_obj.logpdf(data, *fitted_params))
            
            if not np.isfinite(max_loglik):
                return None
            
            # Critical value for confidence interval (chi-squared with 1 df)
            chi2_critical = chi2.ppf(confidence_level, 1)
            loglik_threshold = max_loglik - chi2_critical / 2
            
            # Get the fitted parameter value
            fitted_value = fitted_params[param_index]
            
            # Search for confidence interval bounds
            # This is a simplified approach - a full implementation would be more sophisticated
            search_range = abs(fitted_value) if fitted_value != 0 else 1.0
            lower_bound = fitted_value - 2 * search_range
            upper_bound = fitted_value + 2 * search_range
            
            # Ensure bounds are reasonable for the parameter type
            if param_index == len(fitted_params) - 1:  # Scale parameter (usually positive)
                lower_bound = max(lower_bound, 1e-6)
            
            def profile_loglik(param_val):
                """Calculate log-likelihood with one parameter fixed."""
                try:
                    test_params = list(fitted_params)
                    test_params[param_index] = param_val
                    loglik = np.sum(dist_obj.logpdf(data, *test_params))
                    return loglik if np.isfinite(loglik) else -1e12
                except:
                    return -1e12
            
            # Simple grid search for bounds (could be improved with optimization)
            n_points = 20
            param_values = np.linspace(lower_bound, upper_bound, n_points)
            logliks = [profile_loglik(pval) for pval in param_values]
            
            # Find bounds where log-likelihood drops below threshold
            valid_indices = [i for i, ll in enumerate(logliks) if ll >= loglik_threshold]
            
            if len(valid_indices) < 2:
                return None  # Couldn't establish confidence interval
            
            ci_lower = param_values[min(valid_indices)]
            ci_upper = param_values[max(valid_indices)]
            
            return (ci_lower, ci_upper)
            
        except Exception as exc:
            return None

    @staticmethod
    def _fit_distribution_worker(args):
        """
        Static method for parallel distribution fitting.
        This method is designed to be used with multiprocessing.
        
        Args:
            args: Tuple containing (dist_name, data, features_dict, config_dict)
        
        Returns:
            FitResult object
        """
        dist_name, data, features_dict, config_dict = args
        
        # Reconstruct DistributionFeatures object from dict
        features = None
        if features_dict:
            features = DistributionFeatures(**features_dict)
        
        # Create a temporary selector instance with the same configuration
        selector = UnifiedDistributionSelectorV6(
            robust_mode=config_dict['robust_mode'],
            use_adaptive_strategy=config_dict['use_adaptive_strategy'],
            use_lrt=False,  # Disable LRT for individual fits
            use_cv=config_dict['use_cv'],
            n_jobs=1,  # No nested parallelization
            fit_estimator=str(config_dict.get("fit_estimator", "mle") or "mle"),
            mps_maxiter=int(config_dict.get("mps_maxiter", 250) or 250),
            mps_tol=float(config_dict.get("mps_tol", 1e-6) or 1e-6),
            compute_ad=bool(config_dict.get('compute_ad', False)),
            ad_bootstrap_samples=int(config_dict.get('ad_bootstrap_samples', 0) or 0),
            compute_qq_pp=bool(config_dict.get('compute_qq_pp', False)),
            interval_likelihood=bool(config_dict.get("interval_likelihood", False)),
            interval_delta=float(config_dict.get("interval_delta", 0.0) or 0.0),
            compute_crps=bool(config_dict.get("compute_crps", False)),
            crps_mc_samples=int(config_dict.get("crps_mc_samples", 96) or 96),
            crps_data_subsample=int(config_dict.get("crps_data_subsample", 256) or 256),
            random_state=config_dict.get('random_state', None),
        )
        
        # Fit the single distribution
        return selector._fit_single_distribution(data, dist_name, features)

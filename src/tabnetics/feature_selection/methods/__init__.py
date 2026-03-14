"""Feature selection method implementations.

Submodules
----------
screening : Tier 2 interaction-aware screening (STIR/ReliefF).
filter    : Univariate filter methods (MI, F-test, WMW-AUC, …).
embedded  : Embedded methods (Lasso, SVM-L1, RF, IPSS, …).
pairwise  : Pairwise methods (k-TSP).
wrapper   : Wrapper/iterative-pruning methods.
knockoff  : Copula knockoff selection (DTDCKe).
multiclass: Multiclass-specific methods (OVA, ECOC, NSC, …).
hsic      : HSIC Lasso.
sdr       : SIR/SAVE/PFC sufficient-dimension reduction selectors.
"""

__all__ = [
    "screening",
    "filter",
    "embedded",
    "pairwise",
    "wrapper",
    "knockoff",
    "multiclass",
    "hsic",
    "sdr",
]

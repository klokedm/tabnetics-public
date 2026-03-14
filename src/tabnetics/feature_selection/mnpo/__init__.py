"""MNPO subpackage — portfolio optimization methods extracted from base.py."""

from .portfolio import (
    runtime_race_candidates,
    evaluate_candidate_library,
    extract_portfolio,
    mnpo_aggregate_feature_votes,
    mnpo_select_features,
)
from .oracles import (
    pairwise_pref_from_fold_scores,
    pairwise_pref_from_scalar,
    estimate_oracle_preferences,
    fit_tritrust_weights,
    aggregate_payoff_matrix,
    normalize_vector_01,
    normalized_mutual_info,
    discretize_signal,
    entropy_discrete,
    mutual_information_discrete,
    pid_imin,
    mirror_descent_mnpo,
)
from .consensus import (
    wrapper_refine_subset_score,
    apply_wrapper_refinement,
    build_rank_aggregation_candidate,
)

__all__ = [
    "runtime_race_candidates",
    "evaluate_candidate_library",
    "extract_portfolio",
    "mnpo_aggregate_feature_votes",
    "mnpo_select_features",
    "pairwise_pref_from_fold_scores",
    "pairwise_pref_from_scalar",
    "estimate_oracle_preferences",
    "fit_tritrust_weights",
    "aggregate_payoff_matrix",
    "normalize_vector_01",
    "normalized_mutual_info",
    "discretize_signal",
    "entropy_discrete",
    "mutual_information_discrete",
    "pid_imin",
    "mirror_descent_mnpo",
    "wrapper_refine_subset_score",
    "apply_wrapper_refinement",
    "build_rank_aggregation_candidate",
]

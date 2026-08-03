"""Compatibility helpers for sklearn version drift in parallel and linear models."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from sklearn import __version__ as _SKLEARN_VERSION
from sklearn.linear_model import LogisticRegression

try:
    # sklearn >= 1.3: preserves sklearn config propagation in workers.
    from sklearn.utils.parallel import Parallel, delayed
except Exception as exc:  # pragma: no cover
    from joblib import Parallel, delayed  # type: ignore


def _parse_major_minor(version: str) -> Tuple[int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)", str(version))
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


_SKLEARN_GE_18 = _parse_major_minor(_SKLEARN_VERSION) >= (1, 8)
_MISSING = object()
__tabnetics_execution_isolated_state__ = {
    "_SKLEARN_GE_18": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": ("scikit-learn",),
    },
}
__tabnetics_execution_ephemeral_globals__ = ("_MISSING",)


def normalize_logistic_regression_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize LogisticRegression kwargs across sklearn <1.8 and >=1.8."""

    params = dict(kwargs)
    if not _SKLEARN_GE_18:
        return params

    penalty = params.pop("penalty", _MISSING)
    l1_ratio_explicit = ("l1_ratio" in params) and (params.get("l1_ratio") is not None)

    if penalty is _MISSING:
        params.setdefault("l1_ratio", 0.0)
    elif penalty is None:
        params["C"] = float("inf")
        params.setdefault("l1_ratio", 0.0)
    else:
        penalty_key = str(penalty).strip().lower()
        if penalty_key == "l1":
            params.setdefault("l1_ratio", 1.0)
        elif penalty_key == "l2":
            params.setdefault("l1_ratio", 0.0)
        elif penalty_key == "elasticnet":
            if not l1_ratio_explicit:
                raise ValueError(
                    "penalty='elasticnet' requires l1_ratio when running with sklearn>=1.8"
                )
        elif penalty_key == "none":
            params["C"] = float("inf")
            params.setdefault("l1_ratio", 0.0)
        else:
            # Unknown penalty string: preserve existing behavior.
            params["penalty"] = penalty

    # sklearn 1.8: n_jobs has no effect for LogisticRegression.
    params.pop("n_jobs", None)
    return params


def make_logistic_regression(**kwargs: Any) -> LogisticRegression:
    """Build LogisticRegression with sklearn-version-safe kwargs."""

    return LogisticRegression(**normalize_logistic_regression_kwargs(kwargs))


__all__ = [
    "Parallel",
    "delayed",
    "make_logistic_regression",
    "normalize_logistic_regression_kwargs",
]

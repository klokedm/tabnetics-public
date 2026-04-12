import pytest

from tabnetics.core import compat as compat


def test_normalize_logistic_kwargs_passthrough_for_pre_18(monkeypatch):
    monkeypatch.setattr(compat, "_SKLEARN_GE_18", False)
    raw = {"penalty": "l1", "solver": "saga", "n_jobs": 8, "C": 0.3}

    normalized = compat.normalize_logistic_regression_kwargs(raw)

    assert normalized == raw
    assert normalized is not raw


def test_normalize_logistic_kwargs_maps_penalty_and_drops_n_jobs_for_ge_18(monkeypatch):
    monkeypatch.setattr(compat, "_SKLEARN_GE_18", True)
    raw = {"penalty": "l2", "solver": "lbfgs", "n_jobs": 8, "C": 0.3}

    normalized = compat.normalize_logistic_regression_kwargs(raw)

    assert "penalty" not in normalized
    assert "n_jobs" not in normalized
    assert normalized["C"] == pytest.approx(0.3)
    assert normalized["l1_ratio"] == pytest.approx(0.0)


def test_normalize_logistic_kwargs_requires_l1_ratio_for_elasticnet_ge_18(monkeypatch):
    monkeypatch.setattr(compat, "_SKLEARN_GE_18", True)

    with pytest.raises(ValueError, match="requires l1_ratio"):
        compat.normalize_logistic_regression_kwargs(
            {"penalty": "elasticnet", "solver": "saga", "max_iter": 100}
        )


def test_make_logistic_regression_uses_normalized_kwargs(monkeypatch):
    monkeypatch.setattr(compat, "_SKLEARN_GE_18", True)

    model = compat.make_logistic_regression(
        penalty="l2",
        solver="lbfgs",
        n_jobs=4,
        max_iter=25,
    )

    params = model.get_params()
    assert params["penalty"] in {"l2", "deprecated"}
    assert params["n_jobs"] is None
    assert params["l1_ratio"] == pytest.approx(0.0)

"""S0d: external_feature_scores kwarg on run_pre_split.

The optional dense per-feature score vector (original-index, pre-slice) is the channel
the DIAKRINO soft-rank prefilter consumes (§2.3).  This pins: None is a strict no-op, a
correctly-sized vector is validated + staged, and a wrong-sized vector fails fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabnetics.pipeline.pipeline import DFFSConfig, DistributionFeatureSelectionPipeline


def _tiny_split():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 8))
    y = np.array([0, 1] * 20)
    return X[:28], y[:28], X[28:], y[28:]


def _cfg():
    return DFFSConfig(enabled_methods=("gradient_boosting", "mutual_information", "anova_f"))


def test_none_is_strict_noop():
    Xtr, ytr, Xte, yte = _tiny_split()
    pipe = DistributionFeatureSelectionPipeline(_cfg())
    result = pipe.run_pre_split(Xtr, ytr, Xte, yte, dataset_name="s0d_none", seed=11,
                                external_feature_scores=None)
    assert result is not None
    assert pipe._diakrino_external_feature_scores is None


def test_valid_vector_is_validated_and_staged():
    Xtr, ytr, Xte, yte = _tiny_split()
    scores = np.linspace(0.0, 1.0, Xtr.shape[1])  # length == n_features (8)
    pipe = DistributionFeatureSelectionPipeline(_cfg())
    result = pipe.run_pre_split(Xtr, ytr, Xte, yte, dataset_name="s0d_valid", seed=11,
                                external_feature_scores=scores)
    assert result is not None
    assert pipe._diakrino_external_feature_scores is not None
    assert pipe._diakrino_external_feature_scores.shape == (Xtr.shape[1],)
    assert np.allclose(pipe._diakrino_external_feature_scores, scores)


def test_wrong_length_fails_fast():
    Xtr, ytr, Xte, yte = _tiny_split()
    pipe = DistributionFeatureSelectionPipeline(_cfg())
    with pytest.raises(ValueError, match="external_feature_scores"):
        pipe.run_pre_split(Xtr, ytr, Xte, yte, dataset_name="s0d_bad", seed=11,
                           external_feature_scores=np.zeros(5))  # 5 != 8

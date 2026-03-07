import logging

import numpy as np
import pytest

from tabnetics.feature_selection.mnpo.portfolio import (
    MNPO_ORACLE_COUNT_CAP,
    enforce_oracle_count_cap,
)


def _oracle_dict(n: int):
    return {f"o{i}": np.full((2, 2), 0.5, dtype=float) for i in range(int(n))}


def test_oracle_cap_allows_exactly_cap():
    meta = enforce_oracle_count_cap(_oracle_dict(MNPO_ORACLE_COUNT_CAP))
    assert int(meta["oracle_count"]) == MNPO_ORACLE_COUNT_CAP
    assert bool(meta["oracle_cap_violation"]) is False


def test_oracle_cap_raises_above_cap():
    with pytest.raises(ValueError, match="oracle cap exceeded"):
        enforce_oracle_count_cap(_oracle_dict(MNPO_ORACLE_COUNT_CAP + 1))


def test_oracle_cap_warns_when_approaching_limit(caplog):
    with caplog.at_level(logging.WARNING):
        meta = enforce_oracle_count_cap(_oracle_dict(MNPO_ORACLE_COUNT_CAP - 1))
    assert bool(meta["oracle_cap_warning"]) is True
    assert any("approaching cap" in rec.message for rec in caplog.records)


def test_val9_oracle_bundle_fits_current_cap():
    # Val-9 activates 11 oracles; guard against accidental cap regression.
    assert int(MNPO_ORACLE_COUNT_CAP) >= 11


def test_val9_oracle_bundle_is_accepted():
    meta = enforce_oracle_count_cap(_oracle_dict(11))
    assert bool(meta["oracle_cap_violation"]) is False

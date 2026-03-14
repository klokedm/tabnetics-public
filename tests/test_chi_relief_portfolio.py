"""Tests for the mnpo_chi_relief_extended portfolio set (T-FS-CHI-003/T-FS-REL-003)."""

from tabnetics.benchmarks.runner import FS_METHOD_SETS


def test_portfolio_exists():
    assert "mnpo_chi_relief_extended" in FS_METHOD_SETS


def test_contains_chi_square_and_relieff():
    portfolio = FS_METHOD_SETS["mnpo_chi_relief_extended"]
    assert "chi_square" in portfolio
    assert "relieff" in portfolio


def test_contains_base_methods():
    portfolio = FS_METHOD_SETS["mnpo_chi_relief_extended"]
    base = {"gradient_boosting", "linear_svm", "mutual_information", "anova_f", "mrmr_jmi"}
    assert base.issubset(set(portfolio))

"""S0b: hard opt-in gate for default_enabled=False methods.

The FS method registry runs all non-deprecated methods when enabled_methods=None
(run-all).  Methods that must be opt-in (e.g. the DIAKRINO candidate methods) set
default_enabled=False; this test pins the gate semantics so a future DIAKRINO method
registration can never silently change production selection.
"""

from __future__ import annotations

from tabnetics.feature_selection.registry import (
    METHOD_REGISTRY,
    MethodSpec,
    method_excluded_by_default,
)


def test_only_known_diakrino_methods_are_opt_in():
    """Registering an opt-in (default_enabled=False) method must be deliberate. The pinned
    set is the DIAKRINO candidate selectors (prior / screening-prior / conformal) plus the
    experimental pathway-proxy group-sparse lasso. Any new default_enabled=False method must
    be added here consciously so it can never silently change production selection."""
    opt_in = {k for k, spec in METHOD_REGISTRY.items() if not spec.default_enabled}
    assert opt_in == {
        "diakrino_prior",
        "diakrino_screening_prior",
        "diakrino_conformal_selection",
        "pathway_group_sparse_lasso",
    }


def test_methodspec_field_default_is_true():
    assert MethodSpec(key="x", label="X", fn_name="_x").default_enabled is True


def test_default_enabled_method_never_gated():
    spec = MethodSpec(key="mutual_information", label="MI", fn_name="_mi", default_enabled=True)
    assert method_excluded_by_default(spec, None) is False
    assert method_excluded_by_default(spec, {"something_else"}) is False
    assert method_excluded_by_default(spec, {"mutual_information"}) is False


def test_opt_in_method_excluded_unless_explicitly_enabled():
    spec = MethodSpec(key="diakrino_prior", label="DIAKRINO prior", fn_name="_diakrino", default_enabled=False)
    # run-all default still excludes it -> registering it changes nothing by default
    assert method_excluded_by_default(spec, None) is True
    # excluded when an allow-list is set that omits it
    assert method_excluded_by_default(spec, {"mutual_information", "boruta"}) is True
    # included only when explicitly named
    assert method_excluded_by_default(spec, {"diakrino_prior"}) is False
    assert method_excluded_by_default(spec, {"diakrino_prior", "boruta"}) is False

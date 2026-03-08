"""Validation-1 gate reporting helpers.

Provides strict/quantile side-by-side reporting while keeping strict mode as
the sole promotion decision driver.  Optionally computes reliability-filtered
strict and bootstrap CI gates for advisory diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Mapping, Sequence

from .gates import GateConfig, compute_gate, compute_bootstrap_gate


def build_side_by_side_gate_report(
    *,
    deltas_by_dataset: Mapping[str, Sequence[float]],
    tier_by_dataset: Mapping[str, str],
    strict_config: GateConfig | None = None,
    quantile_config: GateConfig | None = None,
    reliable_config: GateConfig | None = None,
    bootstrap_config: GateConfig | None = None,
    baseline_seed_std_by_dataset: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Compute up to four gate diagnostics side-by-side.

    * **strict** (always computed) — production gate, unchanged.
    * **quantile** (always computed) — advisory trimmed gate.
    * **reliable** (opt-in) — strict gate with reliability filter applied.
    * **bootstrap** (opt-in) — CI-based gate with permutation p-values.

    Promotion decisions remain strict-gate based; all others are advisory only.
    """
    strict_cfg = strict_config or GateConfig(mode="strict")
    quantile_cfg = quantile_config or GateConfig(mode="quantile", easy_trim=1, medium_trim=0)

    strict_res = compute_gate(
        deltas_by_dataset=deltas_by_dataset,
        tier_by_dataset=tier_by_dataset,
        config=strict_cfg,
    )
    quantile_res = compute_gate(
        deltas_by_dataset=deltas_by_dataset,
        tier_by_dataset=tier_by_dataset,
        config=quantile_cfg,
    )

    report: Dict[str, Any] = {
        "primary_mode": "strict",
        "strict": asdict(strict_res),
        "quantile": asdict(quantile_res),
    }

    # Reliability-filtered strict gate (opt-in).
    if reliable_config is not None:
        reliable_res = compute_gate(
            deltas_by_dataset=deltas_by_dataset,
            tier_by_dataset=tier_by_dataset,
            config=reliable_config,
            baseline_seed_std_by_dataset=baseline_seed_std_by_dataset,
        )
        report["reliable"] = asdict(reliable_res)

    # Bootstrap CI gate (opt-in).
    if bootstrap_config is not None:
        bootstrap_res = compute_bootstrap_gate(
            deltas_by_dataset=deltas_by_dataset,
            tier_by_dataset=tier_by_dataset,
            config=bootstrap_config,
            baseline_seed_std_by_dataset=baseline_seed_std_by_dataset,
        )
        report["bootstrap"] = asdict(bootstrap_res)

    # Promotion decision — strict gate is the primary driver.
    report["promotion_decision"] = {
        "mode_used": "strict",
        "verdict": str(strict_res.verdict),
        "promote": bool(strict_res.verdict == "PASS"),
        "advisory_quantile_verdict": str(quantile_res.verdict),
        "advisory_reliable_verdict": (
            str(report["reliable"]["verdict"]) if "reliable" in report else None
        ),
        "advisory_bootstrap_verdict": (
            str(report["bootstrap"]["verdict"]) if "bootstrap" in report else None
        ),
    }
    report["decision_rationale"] = (
        "Strict gate remains the default promotion policy; "
        "quantile, reliability-filtered, and bootstrap gates are reported "
        "side-by-side for sensitivity analysis."
    )

    return report

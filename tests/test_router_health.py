from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tabnetics.validation.router_health import (
    RouterHealthBaseline,
    analyze_router_health,
    join_router_realized_scores,
    load_router_health_baseline,
    merge_descriptor_predictions,
    predict_router_on_descriptors,
    write_router_health_outputs,
)


def _snapshot(
    candidate: str,
    *,
    policy_defaulted: bool = False,
    beats: float = 0.75,
    dependence_policy: str = "",
    dependence_inputs_enabled: bool = False,
    missingness_policy: str = "",
    missingness_inputs_enabled: bool = False,
) -> dict:
    return {
        "auto_router_enabled": True,
        "auto_router_used": True,
        "auto_router_selected_candidate_id": candidate,
        "auto_router_raw_selected_candidate_id": candidate,
        "auto_router_default_candidate_id": "default",
        "auto_router_policy_defaulted": policy_defaulted,
        "auto_router_predicted_balanced_accuracy": 0.72,
        "auto_router_predicted_macro_f1": 0.70,
        "auto_router_predicted_utility": 0.714,
        "auto_router_calibrated_utility": 0.51,
        "auto_router_utility_margin": 0.03,
        "auto_router_beats_default_probability": beats,
        "auto_router_dependence_descriptor_policy": dependence_policy,
        "auto_router_dependence_descriptor_model_input_enabled": dependence_inputs_enabled,
        "auto_router_missingness_descriptor_policy": missingness_policy,
        "auto_router_missingness_descriptor_model_input_enabled": missingness_inputs_enabled,
    }


def _baseline() -> RouterHealthBaseline:
    return RouterHealthBaseline(
        name="test_v25",
        n_policy_groups=10,
        policy_defaulted_rate=0.50,
        non_default_rate=0.30,
        mean_delta_balanced_accuracy_vs_default=0.01,
        mean_delta_macro_f1_vs_default=0.01,
        mean_delta_utility_vs_default=0.01,
        beats_default_probability_threshold=0.5,
        decision_threshold=0.015,
        balanced_accuracy_lcb_offset=0.1,
        macro_f1_lcb_offset=0.1,
    )


def test_router_health_joins_snapshot_rows_and_skips_router_disabled():
    selected = pd.DataFrame(
        [
            {
                "dataset_id": "ds1",
                "seed": 0,
                "protocol": "holdout",
                "config_snapshot": json.dumps(
                    {
                        "auto_router_last_decision": _snapshot(
                            "wide",
                            dependence_policy="bounded_binned_mi_v1",
                            dependence_inputs_enabled=True,
                            missingness_policy="support_mask_summary_v1",
                            missingness_inputs_enabled=True,
                        )
                    }
                ),
                "balanced_accuracy": 0.80,
                "macro_f1": 0.70,
            },
            {
                "dataset_id": "ds2",
                "seed": 0,
                "protocol": "holdout",
                "config_snapshot": json.dumps(
                    {"auto_router_last_decision": {"auto_router_used": False}}
                ),
                "balanced_accuracy": 0.40,
                "macro_f1": 0.40,
            },
            {
                "dataset_id": "plain",
                "seed": 0,
                "protocol": "holdout",
                "balanced_accuracy": 0.99,
                "macro_f1": 0.99,
            },
        ]
    )
    default = pd.DataFrame(
        [
            {
                "dataset_id": "ds1",
                "seed": 0,
                "protocol": "holdout",
                "balanced_accuracy": 0.70,
                "macro_f1": 0.65,
            }
        ]
    )

    joined = join_router_realized_scores(selected, default)

    assert len(joined) == 1
    assert joined.iloc[0]["selected_candidate_id"] == "wide"
    assert bool(joined.iloc[0]["default_arm_available"])
    assert np.isclose(joined.iloc[0]["delta_balanced_accuracy_vs_default"], 0.10)
    assert joined.iloc[0]["auto_router_dependence_descriptor_policy"] == "bounded_binned_mi_v1"
    assert bool(joined.iloc[0]["auto_router_dependence_descriptor_model_input_enabled"])
    assert joined.iloc[0]["auto_router_missingness_descriptor_policy"] == "support_mask_summary_v1"
    assert bool(joined.iloc[0]["auto_router_missingness_descriptor_model_input_enabled"])


def test_router_health_reports_binomial_wilcoxon_and_harvest_rows():
    selected = pd.DataFrame(
        [
            {
                "dataset_id": "ds1",
                "seed": 0,
                "protocol": "holdout",
                "dataset_name": "Dataset 1",
                "domain": "genomics",
                "effective_tier": "hard",
                "auto_router_used": True,
                **_snapshot(
                    "wide",
                    beats=0.80,
                    dependence_policy="bounded_binned_mi_v1",
                    dependence_inputs_enabled=True,
                    missingness_policy="support_mask_summary_v1",
                    missingness_inputs_enabled=True,
                ),
                "balanced_accuracy": 0.80,
                "macro_f1": 0.70,
            },
            {
                "dataset_id": "ds2",
                "seed": 0,
                "protocol": "holdout",
                "auto_router_used": True,
                **_snapshot("default", policy_defaulted=True, beats=0.20),
                "balanced_accuracy": 0.60,
                "macro_f1": 0.60,
            },
        ]
    )
    default = pd.DataFrame(
        [
            {
                "dataset_id": "ds1",
                "seed": 0,
                "protocol": "holdout",
                "balanced_accuracy": 0.70,
                "macro_f1": 0.65,
            },
            {
                "dataset_id": "ds2",
                "seed": 0,
                "protocol": "holdout",
                "balanced_accuracy": 0.62,
                "macro_f1": 0.61,
            },
        ]
    )

    summary, joined, harvest = analyze_router_health(
        selected,
        default,
        baseline=_baseline(),
    )

    assert summary["n_router_rows"] == 2
    assert summary["n_default_arm_rows"] == 2
    assert summary["policy_defaulted_binomial"]["n"] == 2
    assert summary["beats_default_binomial"]["n"] == 2
    assert summary["delta_utility_wilcoxon_vs_baseline"]["center"] == 0.01
    assert len(joined) == 2
    assert len(harvest) == 2
    assert harvest.loc[0, "auto_router_dependence_descriptor_policy"] == "bounded_binned_mi_v1"
    assert bool(harvest.loc[0, "auto_router_dependence_descriptor_model_input_enabled"])
    assert harvest.loc[0, "auto_router_missingness_descriptor_policy"] == "support_mask_summary_v1"
    assert bool(harvest.loc[0, "auto_router_missingness_descriptor_model_input_enabled"])
    assert {
        "dataset_id",
        "candidate_id",
        "balanced_accuracy",
        "default_balanced_accuracy",
        "pred_balanced_accuracy",
        "calibrated_utility",
        "policy_defaulted",
    }.issubset(set(harvest.columns))


def test_router_health_degrades_without_default_arm_and_writes_outputs(tmp_path):
    selected = pd.DataFrame(
        [
            {
                "dataset_id": "ds1",
                "seed": 0,
                "protocol": "holdout",
                "auto_router_used": True,
                **_snapshot("wide"),
                "balanced_accuracy": 0.80,
                "macro_f1": 0.70,
            }
        ]
    )

    paths = write_router_health_outputs(
        selected,
        output_dir=tmp_path,
        baseline=_baseline(),
    )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

    assert summary["n_router_rows"] == 1
    assert summary["n_default_arm_rows"] == 0
    assert summary["beats_default_binomial"]["status"] == "default_arm_missing"
    assert paths["joined"].exists()
    assert paths["harvest"].exists()


def test_router_health_loads_packaged_v25_baseline():
    baseline = load_router_health_baseline()

    assert baseline.n_policy_groups == 513
    assert baseline.policy_defaulted_rate > 0.0
    assert baseline.mean_delta_utility_vs_default > 0.0


def test_router_health_can_run_packaged_router_on_precomputed_descriptors():
    descriptors = pd.DataFrame(
        [
            {
                "dataset_id": "descriptor-smoke",
                "dataset_name": "Descriptor Smoke",
                "seed": 0,
                "log_n": 4.0,
                "log_p": 3.0,
                "log_p_over_n": -1.0,
                "class_count": 2.0,
            }
        ]
    )

    predictions = predict_router_on_descriptors(descriptors)

    assert len(predictions) == 1
    assert bool(predictions.iloc[0]["auto_router_used"])
    assert str(predictions.iloc[0]["auto_router_selected_candidate_id"])


def test_router_health_merges_descriptor_predictions_into_realized_rows():
    realized = pd.DataFrame(
        [
            {
                "dataset_id": "ds",
                "seed": 0,
                "protocol": "holdout",
                "balanced_accuracy": 0.75,
                "macro_f1": 0.70,
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "dataset_id": "ds",
                "seed": 0,
                "protocol": "holdout",
                "auto_router_used": True,
                "auto_router_selected_candidate_id": "wide",
            }
        ]
    )

    merged = merge_descriptor_predictions(realized, predictions)

    assert merged.iloc[0]["auto_router_selected_candidate_id"] == "wide"
    assert bool(merged.iloc[0]["auto_router_used"])

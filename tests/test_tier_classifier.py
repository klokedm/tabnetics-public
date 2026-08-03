import json

import pytest

from tabnetics.datasets.tier_classifier import (
    DEFAULT_VAL20_COMPOSITE_PROFILE_IDS,
    EXPANDED_META_FEATURE_KEYS,
    adaptive_prefilter_top_k,
    adjust_oracle_weights_for_complexity,
    classifier_oracle_shrinkage_factor,
    derive_ground_truth_tier,
    load_tier_classifier_model,
    normalized_complexity_score,
    predict_tier_with_details,
    resolve_composite_training_target,
    train_serialized_tier_classifier,
)


def _meta(
    *,
    n: float,
    p: float,
    class_count: float,
    fisher_f1: float,
    n1_borderline: float,
    intrinsic_dim: float,
    signal_eigenvalue_fraction: float = 0.5,
    f2_overlap: float = 0.2,
) -> dict[str, float]:
    meta = {key: 0.0 for key in EXPANDED_META_FEATURE_KEYS}
    meta.update(
        {
            "n": float(n),
            "p": float(p),
            "p_over_n": float(p) / max(float(n), 1.0),
            "class_count": float(class_count),
            "class_balance_entropy": 0.9,
            "correlation_spectrum_decay": 0.2,
            "heaping_fraction": 0.0,
            "fisher_f1": float(fisher_f1),
            "f2_overlap": float(f2_overlap),
            "n1_borderline": float(n1_borderline),
            "n2_nn_ratio": 0.3,
            "lsc": 0.4,
            "t4_pca_ratio": 0.2,
            "intrinsic_dim": float(intrinsic_dim),
            "correlation_alpha": 0.1,
            "signal_eigenvalue_fraction": float(signal_eigenvalue_fraction),
        }
    )
    return meta


def _row(label: str, *, n: float, p: float, fisher_f1: float, n1: float, intrinsic_dim: float) -> dict[str, object]:
    return {
        "dataset_id": f"{label}_{n}_{p}",
        "label": label,
        "meta_features": _meta(
            n=n,
            p=p,
            class_count=2.0 if label != "very_hard" else 6.0,
            fisher_f1=fisher_f1,
            n1_borderline=n1,
            intrinsic_dim=intrinsic_dim,
            signal_eigenvalue_fraction=0.8 if label in {"hard", "very_hard"} else 0.3,
            f2_overlap=0.7 if label in {"hard", "very_hard"} else 0.1,
        ),
    }


def test_resolve_composite_training_target_requires_winner_profile_id():
    score_map = {
        profile_id: 0.70 + (idx * 0.01)
        for idx, profile_id in enumerate(DEFAULT_VAL20_COMPOSITE_PROFILE_IDS)
    }
    with pytest.raises(ValueError, match="winner unresolved"):
        resolve_composite_training_target(score_map, winner_profile_id=None)


def test_resolve_composite_training_target_computes_oracle_gap():
    score_map = {
        "V20_C01_candidate_a_full64": 0.71,
        "V20_C02_candidate_b_full64": 0.75,
        "V20_C03_candidate_c_full64": 0.72,
        "V20_C04_current_default_full64": 0.69,
    }
    target = resolve_composite_training_target(
        score_map,
        winner_profile_id="V20_C04_current_default_full64",
    )
    assert target["winner_profile_id"] == "V20_C04_current_default_full64"
    assert target["best_composite_profile_id"] == "V20_C02_candidate_b_full64"
    assert target["oracle_gap"] == pytest.approx(0.06)


@pytest.mark.parametrize(
    ("winner_score", "best_score", "n", "classes", "expected"),
    [
        (0.74, 0.742, 100.0, 2.0, "easy"),
        (0.74, 0.750, 100.0, 2.0, "medium"),
        (0.74, 0.770, 100.0, 2.0, "hard"),
        (0.74, 0.860, 48.0, 6.0, "very_hard"),
    ],
)
def test_derive_ground_truth_tier_uses_composite_oracle_gap(
    winner_score: float,
    best_score: float,
    n: float,
    classes: float,
    expected: str,
):
    tier = derive_ground_truth_tier(
        winner_score=winner_score,
        best_composite_score=best_score,
        meta_features=_meta(
            n=n,
            p=400.0,
            class_count=classes,
            fisher_f1=0.4,
            n1_borderline=0.2,
            intrinsic_dim=12.0,
        ),
    )
    assert tier == expected


def test_predict_tier_with_details_falls_back_when_model_is_missing(tmp_path):
    pred = predict_tier_with_details(
        _meta(
            n=120.0,
            p=1200.0,
            class_count=2.0,
            fisher_f1=0.2,
            n1_borderline=0.5,
            intrinsic_dim=20.0,
        ),
        mode="learned",
        model_path=tmp_path / "missing_model.json",
    )
    assert pred.mode == "learned"
    assert pred.fallback_applied is True
    assert pred.model_source == "heuristic_fallback"
    assert pred.tier in {"easy", "medium", "hard", "very_hard"}


def test_learned_tier_model_is_fresh_and_records_exact_bytes(tmp_path):
    model_path = tmp_path / "tier_model.json"
    initial_payload = {
        "required_features": ["n"],
        "tree": {"leaf": {"easy": 1.0}, "prediction": "easy"},
    }
    model_path.write_text(json.dumps(initial_payload), encoding="utf-8")
    loaded = load_tier_classifier_model(model_path)
    loaded["tree"]["prediction"] = "very_hard"

    first = predict_tier_with_details(
        _meta(
            n=120.0,
            p=1200.0,
            class_count=2.0,
            fisher_f1=0.2,
            n1_borderline=0.5,
            intrinsic_dim=20.0,
        ),
        mode="learned",
        model_path=model_path,
    )
    assert first.tier == "easy"
    assert first.model_sha256
    assert first.model_path == str(model_path.resolve())

    replacement_payload = {
        "required_features": ["n"],
        "tree": {"leaf": {"hard": 1.0}, "prediction": "hard"},
    }
    model_path.write_text(json.dumps(replacement_payload), encoding="utf-8")
    replaced = predict_tier_with_details(
        _meta(
            n=120.0,
            p=1200.0,
            class_count=2.0,
            fisher_f1=0.2,
            n1_borderline=0.5,
            intrinsic_dim=20.0,
        ),
        mode="learned",
        model_path=model_path,
    )

    assert replaced.tier == "hard"
    assert replaced.model_sha256 != first.model_sha256
    assert replaced.to_snapshot()["fallback_applied"] is False


def test_complexity_score_and_prefilter_scaling_follow_dataset_difficulty():
    easy_meta = _meta(
        n=180.0,
        p=40.0,
        class_count=2.0,
        fisher_f1=2.0,
        n1_borderline=0.05,
        intrinsic_dim=4.0,
        signal_eigenvalue_fraction=0.2,
        f2_overlap=0.05,
    )
    hard_meta = _meta(
        n=80.0,
        p=8000.0,
        class_count=4.0,
        fisher_f1=0.08,
        n1_borderline=0.62,
        intrinsic_dim=24.0,
        signal_eigenvalue_fraction=0.8,
        f2_overlap=0.7,
    )
    easy_score = normalized_complexity_score(easy_meta)
    hard_score = normalized_complexity_score(hard_meta)
    assert hard_score > easy_score

    easy_top_k = adaptive_prefilter_top_k(
        base_top_k=100,
        n_features=500,
        meta_features=easy_meta,
        scaling_factor=0.5,
    )
    hard_top_k = adaptive_prefilter_top_k(
        base_top_k=100,
        n_features=500,
        meta_features=hard_meta,
        scaling_factor=0.5,
    )
    assert hard_top_k > easy_top_k >= 100


def test_complexity_conditioning_helpers_shift_easy_vs_hard_profiles():
    base_weights = {
        "stability": 1.0,
        "diversity": 1.0,
        "robustness": 1.0,
        "complexity": 1.0,
    }
    easy_meta = _meta(
        n=180.0,
        p=40.0,
        class_count=2.0,
        fisher_f1=2.0,
        n1_borderline=0.05,
        intrinsic_dim=4.0,
    )
    hard_meta = _meta(
        n=80.0,
        p=8000.0,
        class_count=4.0,
        fisher_f1=0.08,
        n1_borderline=0.62,
        intrinsic_dim=24.0,
    )

    easy_weights = adjust_oracle_weights_for_complexity(base_weights, easy_meta)
    hard_weights = adjust_oracle_weights_for_complexity(base_weights, hard_meta)

    assert easy_weights["stability"] > base_weights["stability"]
    assert easy_weights["diversity"] < base_weights["diversity"]
    assert hard_weights["robustness"] > base_weights["robustness"]
    assert hard_weights["complexity"] > base_weights["complexity"]
    assert classifier_oracle_shrinkage_factor(easy_meta) > 1.0
    assert classifier_oracle_shrinkage_factor(hard_meta) < 1.0


def test_complexity_conditioning_helpers_stay_neutral_without_runtime_meta():
    base_weights = {"stability": 0.8, "diversity": 1.1}
    assert adjust_oracle_weights_for_complexity(base_weights, None) == base_weights
    assert classifier_oracle_shrinkage_factor(None) == pytest.approx(1.0)


def test_complexity_conditioning_helpers_handle_empty_weight_map():
    meta = _meta(
        n=64.0,
        p=3200.0,
        class_count=4.0,
        fisher_f1=0.12,
        n1_borderline=0.55,
        intrinsic_dim=18.0,
    )
    assert adjust_oracle_weights_for_complexity({}, meta) == {}


def test_train_serialized_tier_classifier_returns_json_serializable_payload():
    rows = [
        _row("easy", n=180, p=30, fisher_f1=1.8, n1=0.05, intrinsic_dim=4),
        _row("easy", n=160, p=40, fisher_f1=1.6, n1=0.08, intrinsic_dim=5),
        _row("medium", n=140, p=180, fisher_f1=0.7, n1=0.18, intrinsic_dim=12),
        _row("medium", n=130, p=220, fisher_f1=0.6, n1=0.20, intrinsic_dim=14),
        _row("hard", n=90, p=1800, fisher_f1=0.2, n1=0.42, intrinsic_dim=28),
        _row("hard", n=85, p=1600, fisher_f1=0.15, n1=0.40, intrinsic_dim=26),
        _row("very_hard", n=42, p=2400, fisher_f1=0.08, n1=0.62, intrinsic_dim=30),
        _row("very_hard", n=36, p=2200, fisher_f1=0.05, n1=0.68, intrinsic_dim=32),
    ]

    payload = train_serialized_tier_classifier(rows, random_state=7)
    json.dumps(payload)

    assert payload["schema_version"] == 1
    assert payload["training_rows"] == len(rows)
    assert 1 <= len(payload["selected_features"]) <= 5
    assert payload["metrics"]["lodocv_accuracy"] >= 0.0

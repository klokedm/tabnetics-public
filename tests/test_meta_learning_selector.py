import json

import numpy as np

from tabnetics.pipeline.pipeline import (
    ClassificationConfig,
    DFFSConfig,
    DistributionFeatureSelectionPipeline,
)
from tabnetics.benchmarks.runner import FS_METHOD_SETS
from tabnetics.feature_selection.meta_learning import (
    DEFAULT_RECORDS_PATH,
    MetaLearningSelector,
    PROFILE_METHOD_SETS,
    apply_runtime_profile_overlay,
    load_training_records,
    normalize_historical_profile,
)


def _record(dataset_id, best_profile, n, p, class_count, max_feature_variance, ref_score=0.60):
    return {
        "dataset_id": dataset_id,
        "best_profile": best_profile,
        "profile_scores": {
            "a_control": 0.58 if best_profile != "a_control" else 0.83,
            "d_default": 0.59 if best_profile != "d_default" else 0.84,
            "v16_ref": ref_score,
            "v16_multiomics": 0.57 if best_profile != "v16_multiomics" else 0.85,
        },
        "meta_features": {
            "n": float(n),
            "p": float(p),
            "p_over_n": float(p) / float(n),
            "class_count": float(class_count),
            "class_balance_entropy": 0.9,
            "correlation_spectrum_decay": 0.2,
            "heaping_fraction": 0.0,
            "class_balance_ratio": 1.2,
            "class_gini_impurity": 0.5,
            "max_feature_variance": float(max_feature_variance),
        },
    }


def _records_fixture():
    return [
        _record("ds_a1", "a_control", 200, 20, 2, 1.0),
        _record("ds_a2", "a_control", 180, 18, 2, 1.2),
        _record("ds_d1", "d_default", 120, 900, 2, 1.5),
        _record("ds_d2", "d_default", 100, 750, 2, 1.3),
        _record("ds_r1", "v16_ref", 70, 3200, 7, 2.0),
        _record("ds_r2", "v16_ref", 90, 2800, 8, 1.8),
        _record("ds_m1", "v16_multiomics", 140, 4200, 2, 28.0),
        _record("ds_m2", "v16_multiomics", 160, 3800, 2, 30.0),
    ]


def test_committed_meta_learning_records_load():
    records = load_training_records(DEFAULT_RECORDS_PATH)
    assert len(records) > 0
    assert DEFAULT_RECORDS_PATH.exists()


def test_profile_normalization_is_deterministic():
    assert normalize_historical_profile("v14_ipss") == "v16_ref"
    assert normalize_historical_profile("v15_multiomics_adapter") == "v16_multiomics"
    assert normalize_historical_profile("unknown_profile") is None


def test_runtime_profile_method_sets_match_benchmark_definitions():
    assert tuple(PROFILE_METHOD_SETS["a_control"]) == tuple(FS_METHOD_SETS["strict_plus_mrmr"])
    assert tuple(PROFILE_METHOD_SETS["d_default"]) == tuple(FS_METHOD_SETS["mnpo_broad_all"])
    assert tuple(PROFILE_METHOD_SETS["v16_ref"]) == tuple(FS_METHOD_SETS["mnpo_v14_core_plus_ipss"])
    assert tuple(PROFILE_METHOD_SETS["v16_multiomics"]) == tuple(FS_METHOD_SETS["mnpo_v14_core_plus_ipss"])


def test_meta_learning_selector_evaluate_beats_static_v16_ref():
    records = _records_fixture()
    selector = MetaLearningSelector(mode="decision_tree", confidence_threshold=0.55)
    summary = selector.evaluate(records)
    assert summary["n_datasets"] == len(records)
    assert summary["routed_mean_ba"] > summary["static_v16_ref_mean_ba"]


def test_meta_learning_selector_confidence_fallback_routes_to_v16_ref():
    selector = MetaLearningSelector(mode="logistic", confidence_threshold=0.99)
    selector.fit(_records_fixture())
    pred = selector.predict(
        {
            "n": 150.0,
            "p": 400.0,
            "p_over_n": 2.66,
            "class_count": 2.0,
            "class_balance_entropy": 0.9,
            "correlation_spectrum_decay": 0.2,
            "heaping_fraction": 0.0,
            "class_balance_ratio": 1.1,
            "class_gini_impurity": 0.5,
            "max_feature_variance": 2.0,
        }
    )
    assert pred["meta_learning_profile_selected"] == "v16_ref"
    assert pred["meta_learning_fallback_applied"] is True


def test_meta_learning_selector_supports_payload_defined_profiles(tmp_path):
    payload = {
        "schema_version": 2,
        "profile_labels": [
            "V20_C01_candidate_a_full64",
            "V20_C04_current_default_full64",
        ],
        "fallback_profile": "V20_C04_current_default_full64",
        "records": [
            {
                "dataset_id": "ds1",
                "best_profile": "V20_C01_candidate_a_full64",
                "profile_scores": {
                    "V20_C01_candidate_a_full64": 0.81,
                    "V20_C04_current_default_full64": 0.74,
                },
                "meta_features": {
                    "n": 120.0,
                    "p": 400.0,
                    "p_over_n": 3.33,
                    "class_count": 2.0,
                    "class_balance_entropy": 0.9,
                    "correlation_spectrum_decay": 0.2,
                    "heaping_fraction": 0.0,
                    "fisher_f1": 0.8,
                    "f2_overlap": 0.1,
                    "n1_borderline": 0.2,
                    "n2_nn_ratio": 0.3,
                    "lsc": 0.4,
                    "t4_pca_ratio": 0.2,
                    "intrinsic_dim": 12.0,
                    "correlation_alpha": 0.1,
                    "signal_eigenvalue_fraction": 0.5,
                },
            },
            {
                "dataset_id": "ds2",
                "best_profile": "V20_C04_current_default_full64",
                "profile_scores": {
                    "V20_C01_candidate_a_full64": 0.71,
                    "V20_C04_current_default_full64": 0.79,
                },
                "meta_features": {
                    "n": 90.0,
                    "p": 1400.0,
                    "p_over_n": 15.55,
                    "class_count": 4.0,
                    "class_balance_entropy": 0.7,
                    "correlation_spectrum_decay": 0.15,
                    "heaping_fraction": 0.0,
                    "fisher_f1": 0.2,
                    "f2_overlap": 0.5,
                    "n1_borderline": 0.45,
                    "n2_nn_ratio": 0.55,
                    "lsc": 0.3,
                    "t4_pca_ratio": 0.45,
                    "intrinsic_dim": 20.0,
                    "correlation_alpha": 0.2,
                    "signal_eigenvalue_fraction": 0.7,
                },
            },
        ],
    }
    records_path = tmp_path / "meta_learning_payload.json"
    records_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    selector = MetaLearningSelector(
        mode="decision_tree",
        confidence_threshold=0.99,
        records_path=records_path,
    ).fit()
    pred = selector.predict(payload["records"][0]["meta_features"])
    assert pred["meta_learning_fallback_profile"] == "V20_C04_current_default_full64"
    assert "V20_C01_candidate_a_full64" in pred["meta_learning_candidate_profiles"]


def test_apply_runtime_profile_overlay_supports_custom_overlays():
    cfg = DFFSConfig(enabled_methods=("gradient_boosting",))
    apply_runtime_profile_overlay(
        cfg,
        "V20_C02_candidate_b_full64",
        runtime_profile_overlays={
            "V20_C02_candidate_b_full64": {
                "enabled_methods": ["linear_svm", "anova_f"],
                "config_overrides": {"selection_strategy": "legacy_voting"},
                "classification_overrides": {"conformal_enabled": False},
            }
        },
    )
    assert tuple(cfg.enabled_methods) == ("linear_svm", "anova_f")
    assert str(cfg.selection_strategy) == "legacy_voting"
    assert bool(cfg.classification.conformal_enabled) is False


def test_runtime_profile_overlay_updates_pipeline_snapshot(monkeypatch):
    def _fit(self, records=None):
        self.model_ = object()
        self.classes_ = ("a_control", "d_default", "v16_ref", "v16_multiomics")
        self.fitted_ = True
        return self

    def _predict_from_arrays(self, X, y):
        return {
            "meta_learning_profile_selected": "a_control",
            "meta_learning_profile_raw": "a_control",
            "meta_learning_confidence": 0.91,
            "meta_learning_fallback_applied": False,
            "meta_learning_candidate_profiles": list(self.classes_),
        }

    monkeypatch.setattr(MetaLearningSelector, "fit", _fit)
    monkeypatch.setattr(MetaLearningSelector, "predict_from_arrays", _predict_from_arrays)

    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 8))
    y = np.array([0, 1] * 20)
    cfg = DFFSConfig(
        meta_learning_selector_mode="decision_tree",
        enabled_methods=("gradient_boosting", "mutual_information", "anova_f"),
    )
    result = DistributionFeatureSelectionPipeline(cfg).run_pre_split(
        X_train=X[:28],
        y_train=y[:28],
        X_test=X[28:],
        y_test=y[28:],
        dataset_name="meta_learning_snapshot",
        seed=11,
    )
    snapshot = dict(result.config_snapshot)
    assert snapshot["meta_learning_profile_selected"] == "a_control"
    assert snapshot["meta_learning_fallback_applied"] is False
    assert "a_control" in snapshot["meta_learning_candidate_profiles"]


def test_apply_runtime_profile_overlay_sets_multiomics_profile():
    cfg = DFFSConfig()
    apply_runtime_profile_overlay(cfg, "v16_multiomics")
    assert cfg.multiomics_adapter == "split_halves"
    assert cfg.multiomics_integrator == "mb_plsda"
    assert bool(cfg.regime_gating_enabled) is True


def test_apply_runtime_profile_overlay_resets_profile_defining_flags():
    cfg = DFFSConfig(
        classification=ClassificationConfig(conformal_enabled=False),
        prefilter_bh_ttest_enabled=False,
        prefilter_variance_floor_enabled=False,
        fs_mrmr_mi_redundancy_enabled=False,
        fs_fold_preference_mode="logistic",
        fs_use_conformal_efficiency=True,
        fs_conformal_efficiency_method="aps",
        fs_oracle_weight_js_shrinkage=True,
        fs_payoff_shrinkage_kappa=0.15,
        model_cv_runtime_containment_enabled=False,
        stage2_ratio_augmentation_enabled=False,
        fs_oracle_weighting_mode="uniform",
    )
    apply_runtime_profile_overlay(cfg, "v16_ref")
    assert bool(cfg.prefilter_bh_ttest_enabled) is True
    assert bool(cfg.prefilter_variance_floor_enabled) is True
    assert bool(cfg.fs_mrmr_mi_redundancy_enabled) is True
    assert str(cfg.fs_fold_preference_mode) == "vote"
    assert bool(cfg.fs_use_conformal_efficiency) is False
    assert str(cfg.fs_conformal_efficiency_method) == "split"
    assert bool(cfg.fs_oracle_weight_js_shrinkage) is False
    assert float(cfg.fs_payoff_shrinkage_kappa) == 0.0
    assert bool(cfg.model_cv_runtime_containment_enabled) is True
    assert bool(cfg.stage2_ratio_augmentation_enabled) is True
    assert bool(cfg.classifier_conformal_enabled) is True
    assert bool(cfg.classification.runtime_containment_enabled) is True
    assert bool(cfg.classification.stage2_ratio_augmentation_enabled) is True
    assert bool(cfg.classification.conformal_enabled) is True
    assert str(cfg.fs_oracle_weighting_mode) == "banzhaf"
    assert tuple(cfg.enabled_methods) == (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "class_pareto_front",
        "boruta",
        "copula_knockoff",
        "decorrelated_stability",
        "relieff",
        "stability_lasso",
        "rfecv",
        "hsic_lasso",
        "joint_multiclass_support",
        "ipss",
    )

import unittest
import warnings
import signal
import time

import numpy as np
from sklearn.datasets import make_classification

import tabnetics.feature_selection.base as feature_selector_module
from tabnetics.feature_selection import FeatureSelector


def _make_hdlss(seed: int = 13):
    return make_classification(
        n_samples=64,
        n_features=48,
        n_informative=12,
        n_redundant=8,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.55, 0.3, 0.15],
        class_sep=1.0,
        random_state=seed,
    )


def _make_binary_hdlss(seed: int = 29):
    return make_classification(
        n_samples=72,
        n_features=56,
        n_informative=14,
        n_redundant=10,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.05,
        random_state=seed,
    )


def _make_uncorrelated_binary(seed: int = 101):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(80, 60))
    y = np.array([0] * 40 + [1] * 40, dtype=int)
    rng.shuffle(y)
    return X.astype(float), y.astype(int)


def _common_kwargs(seed: int = 13):
    return dict(
        n_bootstrap_iterations=1,
        random_state=seed,
        problem_type="classification",
        enabled_methods=["stability_lasso", "gradient_boosting", "linear_svm", "mutual_information", "anova_f"],
    )


class TestMNPOFeatureSelector(unittest.TestCase):
    def test_mnpo_selector_produces_portfolio_result(self):
        X, y = _make_hdlss(13)
        selector = FeatureSelector(
            **_common_kwargs(13),
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=60,
            portfolio_size=3,
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=12, return_result_object=True)

        self.assertEqual(X_selected.shape, (X.shape[0], 12))
        self.assertEqual(len(result.selected_feature_indices), 12)
        self.assertEqual(result.config["selection_strategy"], "mnpo_portfolio")
        self.assertIn("mnpo_portfolio", result.method_results)
        self.assertTrue(selector.mnpo_diagnostics_.get("portfolio_candidates"))
        self.assertTrue(np.all(result.selected_feature_indices < X.shape[1]))

    def test_legacy_strategy_still_works(self):
        X, y = _make_hdlss(17)
        selector = FeatureSelector(
            **_common_kwargs(17),
            selection_strategy="legacy_voting",
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)

        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertEqual(result.config["selection_strategy"], "legacy_voting")
        self.assertNotIn("mnpo_portfolio", result.method_results)

    def test_variance_filter_keeps_all_features_when_threshold_would_drop_everything(self):
        X, y = _make_binary_hdlss(23)
        X = X * 1e-3
        kwargs = _common_kwargs(23)
        kwargs["enabled_methods"] = ["anova_f", "mutual_information"]
        selector = FeatureSelector(
            **kwargs,
            selection_strategy="legacy_voting",
            variance_threshold=0.01,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            X_selected, result = selector.fit_transform(
                X, y, n_final_features=6, return_result_object=True
            )

        self.assertEqual(X_selected.shape, (X.shape[0], 6))
        self.assertEqual(len(result.selected_feature_indices), 6)
        self.assertEqual(result.eliminated_features["low_variance"], [])
        self.assertTrue(
            any("VarianceThreshold would drop all features" in str(item.message) for item in caught)
        )

    def test_mnpo_oracle_toggles_remove_disabled_oracles(self):
        X, y = _make_hdlss(19)
        selector = FeatureSelector(
            **_common_kwargs(19),
            selection_strategy="mnpo_portfolio",
            use_stability_oracle=True,
            use_complexity_oracle=True,
            use_robust_oracle=False,
            use_diversity_oracle=False,
            use_tritrust=True,
            inner_cv_splits=3,
            inner_cv_repeats=1,
        )

        selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        oracle_weights = selector.mnpo_diagnostics_.get("oracle_weights", {})

        self.assertIn("performance", oracle_weights)
        self.assertIn("stability", oracle_weights)
        self.assertIn("complexity", oracle_weights)
        self.assertNotIn("robustness", oracle_weights)
        self.assertNotIn("diversity", oracle_weights)

    def test_new_methods_smoke_and_emit_results(self):
        X, y = _make_binary_hdlss(29)
        selector = FeatureSelector(
            n_bootstrap_iterations=2,
            random_state=29,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            enabled_methods=[
                "stability_subsample",
                "mrmr_jmi",
                "ktsp",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("stability_subsample", result.method_results)
        self.assertIn("mrmr_jmi", result.method_results)
        self.assertIn("ktsp", result.method_results)
        self.assertTrue(len(result.method_results["stability_subsample"].get("selected_indices", [])) > 0)
        self.assertTrue(len(result.method_results["mrmr_jmi"].get("selected_indices", [])) > 0)
        self.assertTrue(len(result.method_results["ktsp"].get("selected_indices", [])) > 0)

    def test_wmw_auc_binary_filter_runs_and_emits_auc_scores(self):
        X, y = _make_binary_hdlss(113)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=113,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            enabled_methods=[
                "wmw_auc",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=12, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 12))
        self.assertIn("wmw_auc", result.method_results)
        self.assertTrue(len(result.method_results["wmw_auc"].get("selected_indices", [])) > 0)
        self.assertIn("auc_scores", result.method_results["wmw_auc"])

    def test_stability_subsample_loss_guided_validation_emits_metadata(self):
        X, y = _make_binary_hdlss(53)
        selector = FeatureSelector(
            n_bootstrap_iterations=2,
            random_state=53,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            stability_use_loss_guided_validation=True,
            stability_validation_fraction=0.30,
            stability_validation_quantile=0.50,
            stability_validation_min_samples=5,
            enabled_methods=[
                "stability_subsample",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("stability_subsample", result.method_results)
        stability_info = result.method_results["stability_subsample"]
        self.assertTrue(bool(stability_info.get("loss_guided_validation_enabled", False)))
        self.assertTrue(np.isclose(float(stability_info.get("loss_guided_validation_fraction")), 0.30))
        self.assertTrue(np.isclose(float(stability_info.get("loss_guided_validation_quantile")), 0.50))
        self.assertGreaterEqual(int(stability_info.get("n_fit_records", 0)), int(stability_info.get("n_fits", 0)))
        self.assertGreater(int(stability_info.get("n_fit_records", 0)), 0)
        self.assertIn("loss_guided_validation_threshold", stability_info)
        self.assertIn("loss_guided_validation_fallback", stability_info)
        self.assertTrue(bool(result.config.get("stability_use_loss_guided_validation", False)))
        self.assertTrue(np.isclose(float(result.config.get("stability_validation_fraction")), 0.30))
        self.assertTrue(np.isclose(float(result.config.get("stability_validation_quantile")), 0.50))
        self.assertEqual(int(result.config.get("stability_validation_min_samples")), 5)

    def test_cluster_stability_method_smoke_and_emits_cluster_stats(self):
        X, y = _make_binary_hdlss(41)
        selector = FeatureSelector(
            n_bootstrap_iterations=2,
            random_state=41,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            enabled_methods=[
                "cluster_stability",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("cluster_stability", result.method_results)
        cluster_info = result.method_results["cluster_stability"]
        self.assertGreaterEqual(int(cluster_info.get("n_clusters", 0)), 1)
        self.assertGreater(len(cluster_info.get("selected_indices", [])), 0)

    def test_decorrelated_stability_method_smoke_and_emits_stats(self):
        X, y = _make_binary_hdlss(45)
        selector = FeatureSelector(
            n_bootstrap_iterations=2,
            random_state=45,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            decorrelated_stability_eps=1e-3,
            enabled_methods=[
                "decorrelated_stability",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("decorrelated_stability", result.method_results)
        ds_info = result.method_results["decorrelated_stability"]
        self.assertGreater(len(ds_info.get("selected_indices", [])), 0)
        self.assertGreater(float(ds_info.get("decorrelation_condition_number", 0.0)), 0.0)
        self.assertGreaterEqual(int(ds_info.get("decorrelation_rank", 0)), 1)

    def test_decorrelated_stability_correlation_gate_can_skip_when_near_orthogonal(self):
        X, y = _make_uncorrelated_binary(105)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=105,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=30,
            portfolio_size=3,
            decorrelated_stability_min_max_abs_corr=0.99,
            enabled_methods=[
                "decorrelated_stability",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("decorrelated_stability", result.method_results)
        ds_info = result.method_results["decorrelated_stability"]
        self.assertTrue(bool(ds_info.get("decorrelation_gated", False)))
        self.assertEqual(len(ds_info.get("selected_indices", [])), 0)
        self.assertTrue(np.isclose(float(ds_info.get("decorrelation_min_max_abs_corr")), 0.99))

    def test_subspace_stability_method_smoke_and_emits_equivalent_models(self):
        X, y = _make_binary_hdlss(49)
        selector = FeatureSelector(
            n_bootstrap_iterations=2,
            random_state=49,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            enabled_methods=[
                "subspace_stability",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("subspace_stability", result.method_results)
        ss_info = result.method_results["subspace_stability"]
        self.assertGreater(len(ss_info.get("selected_indices", [])), 0)
        self.assertIn("equivalent_models", ss_info)
        self.assertGreaterEqual(len(ss_info.get("equivalent_models", [])), 1)
        self.assertGreaterEqual(int(ss_info.get("n_subspace_groups", 0)), 0)
        self.assertIn("subspace_corr_threshold", ss_info)
        self.assertIn("subspace_group_sizes", ss_info)

    def test_tigress_stability_method_smoke_and_emits_path_stats(self):
        X, y = _make_binary_hdlss(57)
        selector = FeatureSelector(
            n_bootstrap_iterations=2,
            random_state=57,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            enabled_methods=[
                "tigress_stability",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("tigress_stability", result.method_results)
        ts_info = result.method_results["tigress_stability"]
        self.assertGreater(len(ts_info.get("selected_indices", [])), 0)
        self.assertIn("tigress_path_grid", ts_info)
        self.assertIn("tigress_path_fit_counts", ts_info)
        self.assertGreaterEqual(len(ts_info.get("tigress_path_grid", [])), 3)
        self.assertEqual(len(ts_info.get("tigress_path_grid", [])), len(ts_info.get("tigress_path_fit_counts", [])))
        self.assertGreaterEqual(float(ts_info.get("tigress_random_weight_low", 0.0)), 0.0)

    def test_rank_aggregation_modes_add_synthetic_candidate(self):
        X, y = _make_binary_hdlss(63)
        for mode in ("borda", "rra"):
            with self.subTest(mode=mode):
                selector = FeatureSelector(
                    n_bootstrap_iterations=1,
                    random_state=63,
                    problem_type="classification",
                    selection_strategy="mnpo_portfolio",
                    inner_cv_splits=3,
                    inner_cv_repeats=1,
                    mirror_descent_steps=40,
                    portfolio_size=3,
                    rank_aggregation_mode=mode,
                    enabled_methods=[
                        "stability_subsample",
                        "linear_svm",
                        "mutual_information",
                        "anova_f",
                    ],
                )

                _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
                self.assertEqual(result.config["rank_aggregation_mode"], mode)
                self.assertIn(f"rank_aggregate_{mode}", selector.mnpo_diagnostics_.get("candidate_names", []))
                mnpo_summary = result.method_results.get("mnpo_portfolio", {})
                self.assertEqual(mnpo_summary.get("rank_aggregation_mode"), mode)
                self.assertEqual(mnpo_summary.get("rank_aggregation_candidate"), f"rank_aggregate_{mode}")

    def test_ova_ensemble_method_smoke_emits_class_metadata(self):
        X, y = _make_hdlss(69)
        for backend in ("linear_svm_l1", "elastic_net_lr"):
            with self.subTest(backend=backend):
                selector = FeatureSelector(
                    n_bootstrap_iterations=1,
                    random_state=69,
                    problem_type="classification",
                    selection_strategy="mnpo_portfolio",
                    inner_cv_splits=3,
                    inner_cv_repeats=1,
                    mirror_descent_steps=40,
                    portfolio_size=3,
                    ova_negative_ratio=1.5,
                    ova_min_classes=3,
                    ova_linear_backend=backend,
                    enabled_methods=[
                        "ova_ensemble",
                        "linear_svm",
                        "mutual_information",
                        "anova_f",
                    ],
                )

                X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
                self.assertEqual(X_selected.shape, (X.shape[0], 10))
                self.assertIn("ova_ensemble", result.method_results)
                ova_info = result.method_results["ova_ensemble"]
                self.assertGreaterEqual(int(ova_info.get("ova_classes_total", 0)), 3)
                self.assertGreaterEqual(int(ova_info.get("ova_classes_used", 0)), 2)
                self.assertIn("ova_class_selected_indices", ova_info)
                self.assertEqual(str(result.config.get("ova_linear_backend")), backend)
                self.assertTrue(np.isclose(float(result.config.get("ova_negative_ratio")), 1.5))
                self.assertEqual(int(ova_info.get("ova_min_classes", 0)), 3)
                self.assertEqual(int(result.config.get("ova_min_classes", 0)), 3)

    def test_ova_ensemble_class_weighting_downweights_rare_class(self):
        X, y = _make_hdlss(73)
        classes, counts = np.unique(y, return_counts=True)
        count_map = {int(c): int(n) for c, n in zip(classes, counts)}
        rare_cls = min(count_map, key=count_map.get)
        major_cls = max(count_map, key=count_map.get)

        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=73,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            ova_negative_ratio=1.5,
            ova_min_classes=3,
            ova_min_pos_samples=2,
            ova_class_weight_mode="sqrt_pos",
            enabled_methods=[
                "ova_ensemble",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        ova_info = result.method_results.get("ova_ensemble", {})
        weights = ova_info.get("ova_class_weights", {}) or {}

        self.assertIn(int(rare_cls), weights)
        self.assertIn(int(major_cls), weights)
        self.assertLess(float(weights[int(rare_cls)]), float(weights[int(major_cls)]))
        self.assertEqual(str(result.config.get("ova_class_weight_mode")), "sqrt_pos")
        self.assertEqual(int(result.config.get("ova_min_pos_samples", 0)), 2)

    def test_ova_ensemble_class_weighting_upweights_rare_class(self):
        X, y = _make_hdlss(74)
        classes, counts = np.unique(y, return_counts=True)
        count_map = {int(c): int(n) for c, n in zip(classes, counts)}
        rare_cls = min(count_map, key=count_map.get)
        major_cls = max(count_map, key=count_map.get)

        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=74,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            ova_negative_ratio=1.5,
            ova_min_classes=3,
            ova_min_pos_samples=2,
            ova_class_weight_mode="inv_sqrt_pos",
            enabled_methods=[
                "ova_ensemble",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        ova_info = result.method_results.get("ova_ensemble", {})
        weights = ova_info.get("ova_class_weights", {}) or {}

        self.assertIn(int(rare_cls), weights)
        self.assertIn(int(major_cls), weights)
        self.assertGreater(float(weights[int(rare_cls)]), float(weights[int(major_cls)]))
        self.assertEqual(str(result.config.get("ova_class_weight_mode")), "inv_sqrt_pos")
        self.assertEqual(int(result.config.get("ova_min_pos_samples", 0)), 2)

    def test_ova_ensemble_p_norm_aggregation_emits_metadata(self):
        X, y = _make_hdlss(75)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=75,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            ova_negative_ratio=1.5,
            ova_min_classes=3,
            ova_aggregation_mode="p_norm",
            ova_aggregation_p=3.0,
            enabled_methods=[
                "ova_ensemble",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        ova_info = result.method_results.get("ova_ensemble", {})
        self.assertEqual(str(ova_info.get("ova_aggregation_mode")), "p_norm")
        self.assertTrue(np.isclose(float(ova_info.get("ova_aggregation_p")), 3.0))
        self.assertEqual(str(result.config.get("ova_aggregation_mode")), "p_norm")

    def test_ova_ensemble_respects_min_classes_gate(self):
        X, y = _make_hdlss(71)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=71,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            ova_min_classes=5,
            enabled_methods=[
                "ova_ensemble",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("ova_ensemble", result.method_results)
        self.assertEqual(result.method_results["ova_ensemble"], {})
        self.assertEqual(int(result.config.get("ova_min_classes", 0)), 5)

    def test_ecoc_class_aware_method_smoke_emits_task_metadata(self):
        X, y = _make_hdlss(76)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=76,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            ecoc_min_classes=3,
            ecoc_max_ovo_pairs=4,
            ecoc_random_code_bits=2,
            ecoc_class_complexity_weight=1.2,
            enabled_methods=[
                "ecoc_class_aware",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("ecoc_class_aware", result.method_results)
        ecoc_info = result.method_results["ecoc_class_aware"]
        self.assertGreater(len(ecoc_info.get("selected_indices", [])), 0)
        self.assertGreater(int(ecoc_info.get("ecoc_tasks_used", 0)), 0)
        self.assertGreaterEqual(int(ecoc_info.get("ecoc_classes_total", 0)), 3)
        self.assertIn("ecoc_task_metadata", ecoc_info)
        self.assertEqual(int(result.config.get("ecoc_min_classes", 0)), 3)
        self.assertEqual(int(result.config.get("ecoc_random_code_bits", 0)), 2)
        self.assertTrue(np.isclose(float(result.config.get("ecoc_class_complexity_weight", 0.0)), 1.2))

    def test_joint_multiclass_support_method_emits_path_metadata(self):
        X, y = _make_hdlss(77)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=77,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            joint_multiclass_min_classes=3,
            joint_multiclass_max_features=96,
            joint_multiclass_path_grid_size=4,
            joint_multiclass_min_c=0.08,
            joint_multiclass_max_c=1.4,
            joint_multiclass_l1_ratio=0.6,
            joint_multiclass_univariate_blend=0.25,
            enabled_methods=[
                "joint_multiclass_support",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("joint_multiclass_support", result.method_results)
        info = result.method_results["joint_multiclass_support"]
        self.assertGreater(len(info.get("selected_indices", [])), 0)
        self.assertGreater(int(info.get("joint_multiclass_fitted_models", 0)), 0)
        self.assertGreaterEqual(int(info.get("joint_multiclass_classes_total", 0)), 3)
        self.assertIn("joint_multiclass_path_metadata", info)
        self.assertEqual(int(result.config.get("joint_multiclass_path_grid_size", 0)), 4)
        self.assertTrue(np.isclose(float(result.config.get("joint_multiclass_l1_ratio", 0.0)), 0.6))

    def test_dove_class_specific_method_emits_matrix_and_path_metadata(self):
        X, y = _make_hdlss(88)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=88,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            dove_min_classes=3,
            dove_max_pairs_per_class=3,
            dove_path_grid_size=4,
            dove_specificity_weight=0.4,
            dove_minority_boost=0.6,
            enabled_methods=[
                "dove_class_specific",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("dove_class_specific", result.method_results)
        info = result.method_results["dove_class_specific"]
        self.assertGreater(len(info.get("selected_indices", [])), 0)
        self.assertGreater(int(info.get("dove_classes_total", 0)), 2)
        self.assertIn("class_specific_relevance_matrix", info)
        self.assertIn("dove_path_metadata", info)
        self.assertEqual(int(result.config.get("dove_path_grid_size", 0)), 4)
        self.assertTrue(np.isclose(float(result.config.get("dove_specificity_weight", 0.0)), 0.4))

    def test_sparse_multinomial_method_emits_path_metadata(self):
        X, y = _make_hdlss(89)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=89,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            sparse_multinomial_min_classes=3,
            sparse_multinomial_max_features=120,
            sparse_multinomial_path_grid_size=4,
            sparse_multinomial_min_c=0.08,
            sparse_multinomial_max_c=1.4,
            sparse_multinomial_backend="elasticnet",
            sparse_multinomial_l1_ratio=0.6,
            sparse_multinomial_univariate_blend=0.25,
            sparse_multinomial_max_iter=4000,
            enabled_methods=[
                "sparse_multinomial",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("sparse_multinomial", result.method_results)
        info = result.method_results["sparse_multinomial"]
        self.assertGreater(len(info.get("selected_indices", [])), 0)
        self.assertGreater(int(info.get("sparse_multinomial_fitted_models", 0)), 0)
        self.assertIn("sparse_multinomial_path_metadata", info)
        self.assertEqual(str(result.config.get("sparse_multinomial_backend", "")), "elasticnet")
        self.assertTrue(np.isclose(float(result.config.get("sparse_multinomial_l1_ratio", 0.0)), 0.6))

    def test_nearest_shrunken_centroid_method_emits_path_metadata(self):
        X, y = _make_hdlss(91)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=91,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            nsc_shrinkage_grid_size=6,
            nsc_min_classes=3,
            enabled_methods=[
                "nearest_shrunken_centroid",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("nearest_shrunken_centroid", result.method_results)
        info = result.method_results["nearest_shrunken_centroid"]
        self.assertGreater(len(info.get("selected_indices", [])), 0)
        self.assertGreaterEqual(int(info.get("nsc_classes_total", 0)), 3)
        self.assertIn("nsc_path_metadata", info)
        self.assertEqual(int(result.config.get("nsc_shrinkage_grid_size", 0)), 6)
        self.assertEqual(int(result.config.get("nsc_min_classes", 0)), 3)

    def test_nearest_shrunken_centroid_supports_threshold_variants(self):
        X, y = _make_hdlss(190)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=190,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            nsc_shrinkage_grid_size=5,
            nsc_min_classes=3,
            nsc_thresholding_mode="auto",
            nsc_order_quantile=0.80,
            nsc_deep_shrinkage_search=True,
            enabled_methods=[
                "nearest_shrunken_centroid",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        info = result.method_results["nearest_shrunken_centroid"]
        self.assertEqual(str(info.get("nsc_thresholding_mode_requested", "")), "auto")
        self.assertIn(str(info.get("nsc_thresholding_mode", "")), {"soft", "hard", "quantile_hard"})
        modes = {str(row.get("mode", "")) for row in info.get("nsc_path_metadata", [])}
        self.assertTrue({"soft", "hard", "quantile_hard"}.issubset(modes))
        self.assertTrue(bool(result.config.get("nsc_deep_shrinkage_search", False)))
        self.assertTrue(np.isclose(float(result.config.get("nsc_order_quantile", 0.0)), 0.80))

    def test_nearest_shrunken_centroid_applies_class_size_normalization(self):
        X, y = _make_hdlss(196)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=196,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            nsc_shrinkage_grid_size=4,
            nsc_min_classes=3,
            enabled_methods=[
                "nearest_shrunken_centroid",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        info = result.method_results["nearest_shrunken_centroid"]
        mk_meta = dict(info.get("nsc_class_size_normalization", {}))

        classes, counts = np.unique(np.asarray(y), return_counts=True)
        n = float(len(y))
        self.assertTrue(mk_meta, "expected NSC class-size normalization metadata")
        for cls, cnt in zip(classes.tolist(), counts.tolist()):
            key = int(cls)
            expected = float(np.sqrt(max(1e-12, (1.0 / float(cnt)) - (1.0 / n))))
            self.assertIn(key, mk_meta)
            self.assertTrue(np.isclose(float(mk_meta[key]), expected))

    def test_class_pareto_front_method_emits_metadata(self):
        X, y = _make_hdlss(191)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=191,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            class_pareto_min_classes=3,
            class_pareto_top_per_class=20,
            class_pareto_global_fraction=0.30,
            class_pareto_minority_boost=0.70,
            class_pareto_kw_weight=0.20,
            enabled_methods=[
                "class_pareto_front",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("class_pareto_front", result.method_results)
        info = result.method_results["class_pareto_front"]
        self.assertGreater(len(info.get("selected_indices", [])), 0)
        self.assertGreaterEqual(int(info.get("class_pareto_classes_used", 0)), 2)
        self.assertGreaterEqual(int(info.get("class_pareto_front_size", 0)), 1)
        self.assertIn("class_pareto_class_top_hits", info)
        self.assertEqual(int(result.config.get("class_pareto_top_per_class", 0)), 20)
        self.assertTrue(np.isclose(float(result.config.get("class_pareto_kw_weight", 0.0)), 0.20))

    def test_hsic_lasso_method_emits_metadata(self):
        X, y = _make_binary_hdlss(192)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=192,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            hsic_lasso_alpha=0.02,
            hsic_lasso_prefilter_max_features=40,
            hsic_lasso_feature_sigma=0.0,
            hsic_lasso_relevance_blend=0.30,
            hsic_lasso_max_iter=3000,
            enabled_methods=[
                "hsic_lasso",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("hsic_lasso", result.method_results)
        info = result.method_results["hsic_lasso"]
        self.assertGreater(len(info.get("selected_indices", [])), 0)
        self.assertGreater(int(info.get("hsic_lasso_pool_size", 0)), 0)
        self.assertIn("hsic_lasso_nonzero_coefficients", info)
        self.assertEqual(int(result.config.get("hsic_lasso_prefilter_max_features", 0)), 40)
        self.assertTrue(np.isclose(float(result.config.get("hsic_lasso_alpha", 0.0)), 0.02))

    def test_runtime_racing_metadata_emits_when_enabled(self):
        X, y = _make_hdlss(90)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=90,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=4,
            runtime_racing_enabled=True,
            runtime_racing_proxy_splits=1,
            runtime_racing_keep_fraction=0.5,
            runtime_racing_min_candidates=2,
            runtime_racing_runtime_weight=0.2,
            enabled_methods=[
                "gradient_boosting",
                "linear_svm",
                "mutual_information",
                "anova_f",
                "mrmr_jmi",
                "ova_ensemble",
                "joint_multiclass_support",
            ],
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertIn("mnpo_portfolio", result.method_results)
        mnpo_info = result.method_results["mnpo_portfolio"]
        self.assertTrue(bool(mnpo_info.get("runtime_racing_enabled", False)))
        self.assertGreaterEqual(int(mnpo_info.get("runtime_racing_initial_candidates", 0)), 2)
        self.assertGreaterEqual(int(mnpo_info.get("runtime_racing_kept_candidates", 0)), 2)
        self.assertLessEqual(
            int(mnpo_info.get("runtime_racing_kept_candidates", 0)),
            int(mnpo_info.get("runtime_racing_initial_candidates", 0)),
        )

    def test_runtime_racing_successive_halving_metadata_emits_when_enabled(self):
        X, y = _make_hdlss(97)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=97,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=4,
            runtime_racing_enabled=True,
            runtime_racing_proxy_splits=2,
            runtime_racing_keep_fraction=0.5,
            runtime_racing_min_candidates=2,
            runtime_racing_runtime_weight=0.15,
            runtime_racing_mode="successive_halving",
            runtime_racing_stages=3,
            runtime_racing_confidence_bound="hoeffding",
            runtime_racing_delta=0.08,
            enabled_methods=[
                "gradient_boosting",
                "linear_svm",
                "mutual_information",
                "anova_f",
                "mrmr_jmi",
                "ova_ensemble",
                "joint_multiclass_support",
            ],
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        mnpo_info = result.method_results["mnpo_portfolio"]
        self.assertEqual(str(mnpo_info.get("runtime_racing_mode", "")), "successive_halving")
        self.assertEqual(str(mnpo_info.get("runtime_racing_confidence_bound", "")), "hoeffding")
        self.assertTrue(np.isclose(float(mnpo_info.get("runtime_racing_delta", 0.0)), 0.08))
        self.assertGreaterEqual(int(mnpo_info.get("runtime_racing_stages", 0)), 1)
        self.assertIn("runtime_racing_stage_history", mnpo_info)
        self.assertTrue(len(mnpo_info.get("runtime_racing_stage_history", [])) >= 1)

    def test_sparse_multinomial_screening_metadata_emits_when_enabled(self):
        X, y = _make_hdlss(98)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=98,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            sparse_multinomial_max_features=40,
            sparse_multinomial_path_grid_size=4,
            sparse_multinomial_backend="elasticnet",
            sparse_multinomial_l1_ratio=0.6,
            sparse_multinomial_screening_mode="prefilter_aggressive",
            sparse_multinomial_screening_keep_fraction=0.5,
            sparse_multinomial_screening_min_features=16,
            sparse_multinomial_screening_fallback_on_failure=True,
            enabled_methods=[
                "sparse_multinomial",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertIn("sparse_multinomial", result.method_results)
        info = result.method_results["sparse_multinomial"]
        self.assertEqual(
            str(info.get("sparse_multinomial_screening_mode", "")),
            "prefilter_aggressive",
        )
        self.assertEqual(
            str(info.get("sparse_multinomial_screening_rule_family", "")),
            "strong_rule_surrogate",
        )
        self.assertIn("sparse_multinomial_screening_applied", info)
        self.assertIn("sparse_multinomial_screening_status", info)
        self.assertIn("sparse_multinomial_screening_attempt", info)
        self.assertIn("sparse_multinomial_screening_score_threshold", info)
        self.assertIn("sparse_multinomial_screening_safety_floor", info)

    def test_sparse_multinomial_screening_modes_emit_distinct_rule_families(self):
        X, y = _make_hdlss(198)

        def _run_mode(mode: str):
            selector = FeatureSelector(
                n_bootstrap_iterations=1,
                random_state=198,
                problem_type="classification",
                selection_strategy="mnpo_portfolio",
                inner_cv_splits=3,
                inner_cv_repeats=1,
                mirror_descent_steps=40,
                portfolio_size=3,
                sparse_multinomial_max_features=48,
                sparse_multinomial_path_grid_size=3,
                sparse_multinomial_backend="elasticnet",
                sparse_multinomial_l1_ratio=0.6,
                sparse_multinomial_screening_mode=mode,
                sparse_multinomial_screening_keep_fraction=0.50,
                sparse_multinomial_screening_min_features=12,
                enabled_methods=[
                    "sparse_multinomial",
                    "linear_svm",
                    "mutual_information",
                    "anova_f",
                ],
            )
            _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
            info = result.method_results["sparse_multinomial"]
            return (
                int(info.get("sparse_multinomial_screening_retained_pool_size", 0)),
                str(info.get("sparse_multinomial_screening_rule_family", "")),
                float(info.get("sparse_multinomial_screening_score_threshold", 0.0)),
            )

        agg_retained, agg_family, agg_thr = _run_mode("prefilter_aggressive")
        bal_retained, bal_family, bal_thr = _run_mode("prefilter_balanced")
        con_retained, con_family, con_thr = _run_mode("prefilter_conservative")

        self.assertEqual(agg_family, "strong_rule_surrogate")
        self.assertEqual(bal_family, "gap_safe_surrogate")
        self.assertEqual(con_family, "slores_surrogate")
        self.assertLessEqual(agg_retained, bal_retained)
        self.assertLessEqual(bal_retained, con_retained)
        self.assertTrue(np.isfinite(agg_thr))
        self.assertTrue(np.isfinite(bal_thr))
        self.assertTrue(np.isfinite(con_thr))

    def test_sparse_multinomial_screening_legacy_alias_maps_with_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selector = FeatureSelector(
                n_bootstrap_iterations=1,
                random_state=188,
                problem_type="classification",
                selection_strategy="mnpo_portfolio",
                sparse_multinomial_screening_mode="strong",
                enabled_methods=["sparse_multinomial"],
            )
        self.assertEqual(
            str(selector.sparse_multinomial_screening_mode),
            "prefilter_aggressive",
        )
        self.assertTrue(
            any("deprecated" in str(w.message).lower() for w in caught),
            "Expected a deprecation warning for legacy screening alias",
        )

    def test_class_pareto_per_class_quota_metadata_emits_when_enabled(self):
        X, y = _make_hdlss(99)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=99,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            per_class_quota_enabled=True,
            per_class_quota_min_per_class=2,
            per_class_quota_max_fraction=0.5,
            enabled_methods=[
                "class_pareto_front",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertIn("class_pareto_front", result.method_results)
        info = result.method_results["class_pareto_front"]
        self.assertTrue(bool(info.get("class_pareto_per_class_quota_enabled", False)))
        self.assertIn("class_pareto_per_class_quota_applied", info)
        self.assertIn("class_pareto_per_class_quota_meta", info)

    def test_ova_calibration_metadata_emits_when_enabled(self):
        X, y = _make_hdlss(100)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=100,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            ova_min_classes=3,
            ova_enable_calibration=True,
            ova_calibration_cv=3,
            enabled_methods=[
                "ova_ensemble",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertIn("ova_ensemble", result.method_results)
        info = result.method_results["ova_ensemble"]
        self.assertTrue(bool(info.get("ova_enable_calibration", False)))
        self.assertEqual(int(info.get("ova_calibration_cv", 0)), 3)
        self.assertIn("ova_class_calibration_reliability", info)
        self.assertTrue(len(info.get("ova_class_calibration_reliability", {})) >= 1)

    def test_iterative_redundancy_pruning_method_emits_metadata(self):
        X, y = _make_binary_hdlss(78)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=78,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            iterative_pruning_pool_factor=2.0,
            iterative_pruning_max_rounds=12,
            iterative_pruning_min_improvement=-0.01,
            iterative_pruning_redundancy_weight=0.70,
            enabled_methods=[
                "iterative_redundancy_pruning",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("iterative_redundancy_pruning", result.method_results)
        prune_info = result.method_results["iterative_redundancy_pruning"]
        self.assertGreater(len(prune_info.get("selected_indices", [])), 0)
        self.assertIn("iterative_pruning_corr_estimator", prune_info)
        self.assertIn(str(prune_info.get("iterative_pruning_corr_estimator")), {"ledoit_wolf", "corrcoef", "corrcoef_fallback", "degenerate"})
        self.assertIn("iterative_pruning_max_cumulative_loss", prune_info)
        self.assertIn("iterative_pruning_cumulative_delta", prune_info)
        self.assertGreaterEqual(
            int(prune_info.get("iterative_pruning_initial_size", 0)),
            int(prune_info.get("iterative_pruning_final_size", 0)),
        )
        self.assertGreater(int(prune_info.get("iterative_pruning_evaluations", 0)), 0)
        self.assertEqual(int(result.config.get("iterative_pruning_max_rounds", 0)), 12)
        self.assertTrue(np.isclose(float(result.config.get("iterative_pruning_pool_factor", 0.0)), 2.0))

    def test_iterative_pruning_cumulative_loss_budget_can_stop_pruning(self):
        X, y = _make_binary_hdlss(91)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=91,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            iterative_pruning_pool_factor=2.0,
            iterative_pruning_max_rounds=20,
            iterative_pruning_min_improvement=-0.02,
            iterative_pruning_max_cumulative_loss=0.015,
            iterative_pruning_redundancy_weight=0.70,
            enabled_methods=[
                "iterative_redundancy_pruning",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        # Patch wrapper scoring to deterministically penalize removals so cumulative-loss budget is exercised.
        def _fake_score(_X_pool, _y_arr, subset_idx):
            return 0.01 * float(len(np.asarray(subset_idx).ravel()))

        selector._wrapper_refine_subset_score = _fake_score  # type: ignore[attr-defined]

        results, _ = selector._iterative_redundancy_pruning_core(
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=int),
            n_target_features=10,
            runtime_bounded=False,
        )
        self.assertEqual(str(results.get("iterative_pruning_stop_reason", "")), "cumulative_loss_budget_exhausted")
        self.assertTrue(np.isfinite(float(results.get("iterative_pruning_cumulative_delta", float("nan")))))
        self.assertGreaterEqual(float(results.get("iterative_pruning_max_cumulative_loss", 0.0)), 0.015)
        self.assertGreaterEqual(float(results.get("iterative_pruning_cumulative_delta", 0.0)), -0.015 - 1e-12)

    def test_iterative_redundancy_pruning_bounded_emits_runtime_budget_metadata(self):
        X, y = _make_hdlss(79)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=79,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            iterative_pruning_pool_factor=2.0,
            iterative_pruning_max_rounds=12,
            iterative_pruning_min_improvement=-0.01,
            iterative_pruning_redundancy_weight=0.7,
            iterative_pruning_bounded_prefilter_cap=90,
            iterative_pruning_bounded_candidate_fraction=0.30,
            iterative_pruning_bounded_min_candidates=2,
            iterative_pruning_bounded_max_evaluations=10,
            iterative_pruning_bounded_max_runtime_seconds=10.0,
            enabled_methods=[
                "iterative_redundancy_pruning_bounded",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("iterative_redundancy_pruning_bounded", result.method_results)
        prune_info = result.method_results["iterative_redundancy_pruning_bounded"]
        self.assertGreater(len(prune_info.get("selected_indices", [])), 0)
        self.assertTrue(bool(prune_info.get("iterative_pruning_runtime_bounded", False)))
        self.assertLessEqual(int(prune_info.get("iterative_pruning_evaluations", 0)), 10)
        self.assertIn("iterative_pruning_candidate_budgets", prune_info)
        self.assertEqual(int(prune_info.get("iterative_pruning_bounded_max_evaluations", 0)), 10)
        self.assertEqual(int(result.config.get("iterative_pruning_bounded_prefilter_cap", 0)), 90)

    def test_iterative_pruning_bounded_cpss_overlay_emits_metadata(self):
        X, y = _make_hdlss(82)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=82,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            iterative_pruning_bounded_use_cpss_overlay=True,
            iterative_pruning_bounded_cpss_pairs=3,
            iterative_pruning_bounded_cpss_stability_threshold=0.55,
            iterative_pruning_bounded_cpss_min_stable_features=2,
            iterative_pruning_bounded_cpss_min_jaccard=0.20,
            iterative_pruning_bounded_cpss_max_score_drop=0.02,
            enabled_methods=[
                "iterative_redundancy_pruning_bounded",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        prune_info = result.method_results["iterative_redundancy_pruning_bounded"]
        self.assertTrue(bool(prune_info.get("iterative_pruning_cpss_overlay_enabled", False)))
        self.assertIn("iterative_pruning_cpss_switch_reason", prune_info)
        self.assertIn("iterative_pruning_cpss_stable_feature_count", prune_info)
        self.assertIn("iterative_pruning_cpss_overlap_jaccard", prune_info)
        self.assertEqual(int(result.config.get("iterative_pruning_bounded_cpss_pairs", 0)), 3)
        self.assertTrue(
            np.isclose(float(result.config.get("iterative_pruning_bounded_cpss_stability_threshold", 0.0)), 0.55)
        )

    def test_iterative_pruning_class_pareto_prefilter_emits_metadata(self):
        X, y = _make_hdlss(83)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=83,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            iterative_pruning_class_pareto_prefilter_enabled=True,
            iterative_pruning_class_pareto_min_classes=3,
            iterative_pruning_class_pareto_top_per_class=28,
            iterative_pruning_class_pareto_global_fraction=0.35,
            iterative_pruning_class_pareto_minority_boost=0.8,
            enabled_methods=[
                "iterative_redundancy_pruning_bounded",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        prune_info = result.method_results["iterative_redundancy_pruning_bounded"]
        self.assertTrue(bool(prune_info.get("iterative_pruning_pareto_prefilter_enabled", False)))
        self.assertTrue(bool(prune_info.get("iterative_pruning_pareto_prefilter_applied", False)))
        self.assertGreaterEqual(int(prune_info.get("iterative_pruning_pareto_prefilter_front_size", 0)), 1)
        self.assertEqual(int(result.config.get("iterative_pruning_class_pareto_min_classes", 0)), 3)
        self.assertTrue(
            np.isclose(float(result.config.get("iterative_pruning_class_pareto_global_fraction", 0.0)), 0.35)
        )

    def test_iterative_pruning_class_pareto_stability_gate_emits_metadata(self):
        X, y = _make_hdlss(84)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=84,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            iterative_pruning_class_pareto_prefilter_enabled=True,
            iterative_pruning_class_pareto_min_classes=3,
            iterative_pruning_class_pareto_top_per_class=24,
            iterative_pruning_class_pareto_stability_gate_enabled=True,
            iterative_pruning_class_pareto_stability_subsamples=4,
            iterative_pruning_class_pareto_stability_fraction=0.70,
            iterative_pruning_class_pareto_stability_threshold=0.55,
            iterative_pruning_class_pareto_stability_min_overlap=0.40,
            iterative_pruning_class_pareto_stability_min_stable_features=2,
            enabled_methods=[
                "iterative_redundancy_pruning_bounded",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        prune_info = result.method_results["iterative_redundancy_pruning_bounded"]
        self.assertTrue(bool(prune_info.get("iterative_pruning_pareto_stability_gate_enabled", False)))
        self.assertTrue(bool(prune_info.get("iterative_pruning_pareto_stability_gate_applied", False)))
        self.assertIn("iterative_pruning_pareto_stability_gate_reason", prune_info)
        self.assertIn("iterative_pruning_pareto_stability_stable_feature_count", prune_info)
        self.assertEqual(int(result.config.get("iterative_pruning_class_pareto_stability_subsamples", 0)), 4)
        self.assertTrue(
            np.isclose(float(result.config.get("iterative_pruning_class_pareto_stability_min_overlap", 0.0)), 0.40)
        )

    def test_wrapper_refinement_emits_mnpo_metadata(self):
        X, y = _make_binary_hdlss(65)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=65,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            wrapper_refine_enabled=True,
            wrapper_refine_top_k=10,
            wrapper_refine_max_add=5,
            wrapper_refine_min_gain=0.0,
            enabled_methods=[
                "stability_subsample",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        mnpo_summary = result.method_results.get("mnpo_portfolio", {})
        wrapper_meta = mnpo_summary.get("wrapper_refinement", {})

        self.assertTrue(bool(wrapper_meta.get("wrapper_refine_enabled", False)))
        self.assertIn("wrapper_refine_pool_size", wrapper_meta)
        self.assertIn("wrapper_refine_evaluations", wrapper_meta)
        self.assertIn("wrapper_refine_stop_reason", wrapper_meta)
        self.assertTrue(bool(result.config.get("wrapper_refine_enabled", False)))
        self.assertEqual(int(result.config.get("wrapper_refine_top_k")), 10)
        self.assertEqual(int(result.config.get("wrapper_refine_max_add")), 5)
        self.assertTrue(np.isclose(float(result.config.get("wrapper_refine_min_gain")), 0.0))

    def test_ipss_method_smoke_emits_qvalues_and_path_stats(self):
        X, y = _make_binary_hdlss(61)
        selector = FeatureSelector(
            n_bootstrap_iterations=2,
            random_state=61,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=40,
            portfolio_size=3,
            ipss_path_grid_size=5,
            ipss_target_fdr=0.2,
            ipss_null_shuffle_rounds=1,
            enabled_methods=[
                "ipss",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("ipss", result.method_results)
        ipss_info = result.method_results["ipss"]
        self.assertGreater(len(ipss_info.get("selected_indices", [])), 0)
        self.assertEqual(len(ipss_info.get("q_values", [])), X.shape[1])
        self.assertEqual(len(ipss_info.get("path_grid", [])), 5)
        self.assertEqual(len(ipss_info.get("path_fit_counts", [])), 5)

    def test_ipss_gate_can_skip_on_binary_when_min_classes_set(self):
        X, y = _make_binary_hdlss(81)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=81,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=30,
            portfolio_size=3,
            ipss_gate_min_classes=3,
            enabled_methods=[
                "ipss",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        self.assertEqual(X_selected.shape, (X.shape[0], 10))
        self.assertIn("ipss", result.method_results)
        ipss_info = result.method_results["ipss"]
        self.assertTrue(bool(ipss_info.get("ipss_gated", False)))
        self.assertEqual(len(ipss_info.get("selected_indices", [])), 0)
        self.assertEqual(int(ipss_info.get("ipss_gate_min_classes", 0)), 3)

    def test_ipss_eats_threshold_respects_minimum_threshold(self):
        X, y = _make_binary_hdlss(67)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=67,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=30,
            portfolio_size=3,
            ipss_path_grid_size=4,
            ipss_use_eats_threshold=True,
            ipss_eats_min_threshold=0.5,
            enabled_methods=[
                "ipss",
                "linear_svm",
                "mutual_information",
                "anova_f",
            ],
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        ipss_info = result.method_results["ipss"]
        self.assertTrue(bool(ipss_info.get("ipss_use_eats_threshold", False)))
        self.assertGreaterEqual(float(ipss_info.get("stable_threshold", 0.0)), 0.5 - 1e-12)
        self.assertIn("eats_exclusion_floor", ipss_info)

    def test_portfolio_size_guard_raise_errors_when_too_small(self):
        with self.assertRaises(ValueError):
            FeatureSelector(
                n_bootstrap_iterations=1,
                random_state=91,
                problem_type="classification",
                selection_strategy="mnpo_portfolio",
                portfolio_size=2,
                portfolio_size_guard="raise",
                enabled_methods=["linear_svm", "mutual_information", "anova_f"],
            )

    def test_portfolio_size_guard_warn_autocorrects_and_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selector = FeatureSelector(
                n_bootstrap_iterations=1,
                random_state=92,
                problem_type="classification",
                selection_strategy="mnpo_portfolio",
                portfolio_size=2,
                portfolio_size_guard="warn",
                enabled_methods=["linear_svm", "mutual_information", "anova_f"],
            )

        self.assertEqual(selector.portfolio_size, 3)
        self.assertTrue(
            any("portfolio_size=2 is smaller than enabled_methods=3" in str(w.message) for w in caught)
        )

    def test_portfolio_size_guard_warn_bumps_adaptive_bounds(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selector = FeatureSelector(
                n_bootstrap_iterations=1,
                random_state=93,
                problem_type="classification",
                selection_strategy="mnpo_portfolio",
                portfolio_size=2,
                portfolio_size_guard="warn",
                enabled_methods=["linear_svm", "mutual_information", "anova_f", "mrmr_jmi"],
                adaptive_portfolio_sizing_enabled=True,
                adaptive_size_min=1,
                adaptive_size_max=2,
            )

        self.assertEqual(selector.portfolio_size, 4)
        self.assertEqual(selector.adaptive_size_max, 4)
        self.assertEqual(selector.adaptive_size_min, 1)
        self.assertTrue(
            any("portfolio_size=2 is smaller than enabled_methods=4" in str(w.message) for w in caught)
        )

    def test_mnpo_consensus_exclude_methods_can_drop_core_candidates(self):
        X, y = _make_binary_hdlss(97)
        selector = FeatureSelector(
            n_bootstrap_iterations=1,
            random_state=97,
            problem_type="classification",
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=30,
            portfolio_size=7,
            mnpo_consensus_exclude_methods=["gradient_boosting", "linear_svm"],
            mnpo_consensus_exclude_protect_top_k=0,
            enabled_methods=[
                "gradient_boosting",
                "linear_svm",
                "mutual_information",
                "anova_f",
                "mrmr_jmi",
            ],
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        mnpo_summary = result.method_results.get("mnpo_portfolio", {})
        portfolio = set(mnpo_summary.get("portfolio_candidates", []) or [])
        self.assertNotIn("gradient_boosting", portfolio)
        self.assertNotIn("linear_svm", portfolio)
        excluded = set(mnpo_summary.get("mnpo_consensus_excluded_methods", []) or [])
        self.assertIn("gradient_boosting", excluded)
        self.assertIn("linear_svm", excluded)

    def test_mi_redundancy_diversity_mode_wires_into_mnpo(self):
        X, y = _make_hdlss(43)
        selector = FeatureSelector(
            **_common_kwargs(43),
            selection_strategy="mnpo_portfolio",
            use_diversity_oracle=True,
            diversity_oracle_mode="mi_redundancy",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=60,
            portfolio_size=3,
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        oracle_weights = selector.mnpo_diagnostics_.get("oracle_weights", {})
        oracle_components = selector.mnpo_diagnostics_.get("oracle_components", {})
        self.assertIn("diversity", oracle_weights)
        self.assertEqual(result.config["diversity_oracle_mode"], "mi_redundancy")
        self.assertIn("diversity", oracle_components)
        self.assertIn("relevance", oracle_components["diversity"])
        self.assertIn("redundancy", oracle_components["diversity"])
        self.assertIn("complementarity", oracle_components["diversity"])

    def test_pid_mi_diversity_mode_wires_into_mnpo(self):
        X, y = _make_hdlss(44)
        selector = FeatureSelector(
            **_common_kwargs(44),
            selection_strategy="mnpo_portfolio",
            use_diversity_oracle=True,
            diversity_oracle_mode="pid_mi",
            inner_cv_splits=3,
            inner_cv_repeats=1,
            mirror_descent_steps=60,
            portfolio_size=3,
        )

        _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
        oracle_weights = selector.mnpo_diagnostics_.get("oracle_weights", {})
        oracle_components = selector.mnpo_diagnostics_.get("oracle_components", {})
        self.assertIn("diversity", oracle_weights)
        self.assertEqual(result.config["diversity_oracle_mode"], "pid_mi")
        self.assertIn("diversity", oracle_components)
        self.assertIn("relevance", oracle_components["diversity"])
        self.assertIn("redundancy", oracle_components["diversity"])
        self.assertIn("complementarity", oracle_components["diversity"])

    def test_adaptive_performance_weights_shift_toward_macro_f1_for_imbalanced_multiclass(self):
        X, y = _make_hdlss(47)
        selector = FeatureSelector(
            **_common_kwargs(47),
            performance_use_adaptive_imbalance=True,
            performance_imbalance_ratio_trigger=1.5,
            performance_balanced_weight=0.6,
            performance_macro_f1_weight=0.4,
        )

        w_bal, w_f1 = selector._resolve_performance_weights(y)
        self.assertLess(w_bal, 0.6)
        self.assertGreater(w_f1, 0.4)

    def test_inner_cv_splits_fallback_when_a_class_has_single_sample(self):
        rng = np.random.default_rng(53)
        X = rng.normal(size=(18, 10))
        y = np.array([0] * 9 + [1] * 8 + [2], dtype=int)  # min class count = 1
        selector = FeatureSelector(
            **_common_kwargs(53),
            selection_strategy="mnpo_portfolio",
            inner_cv_splits=3,
            inner_cv_repeats=1,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            splits = selector._get_inner_cv_splits(X, y)

        self.assertGreaterEqual(len(splits), 2)
        self.assertFalse(
            any("least populated class" in str(w.message) for w in caught),
            "Stratified split warning should be avoided via class-sparse fallback.",
        )

    def test_fit_and_score_fold_suppresses_label_support_warning(self):
        rng = np.random.default_rng(59)
        X_train = rng.normal(size=(18, 6))
        y_train = np.array([0] * 6 + [1] * 6 + [2] * 6, dtype=int)
        X_val = rng.normal(size=(6, 6))
        y_val = np.array([0, 0, 0, 1, 1, 1], dtype=int)  # class 2 intentionally absent

        selector = FeatureSelector(
            **_common_kwargs(59),
            selection_strategy="mnpo_portfolio",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score, signal = selector._fit_and_score_fold(X_train, y_train, X_val, y_val)

        self.assertTrue(np.isfinite(score))
        self.assertEqual(signal.shape[0], y_val.shape[0])
        self.assertFalse(
            any("y_pred contains classes not in y_true" in str(w.message) for w in caught),
            "Fold scoring should suppress label-support warnings for sparse validation folds.",
        )


def test_copula_knockoff_controls_are_forwarded(monkeypatch):
    X, y = _make_binary_hdlss(73)
    captured = {}

    class _StubCopulaSelector:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.truncation_level_effective_ = {"vine_x": 2, "vine_2p": 2}

        def fit(self, X_fit, y_fit):
            captured["fit_shape"] = tuple(X_fit.shape)
            captured["fit_y_shape"] = tuple(np.asarray(y_fit).shape)
            return self

        def get_weights(self):
            return np.linspace(1.0, 0.1, num=X.shape[1], dtype=float)

        def get_support(self):
            return np.array([0, 1, 2, 3], dtype=int)

    monkeypatch.setattr(feature_selector_module, "CopulaKnockoffSelector", _StubCopulaSelector)

    selector = FeatureSelector(
        n_bootstrap_iterations=1,
        random_state=73,
        problem_type="classification",
        selection_strategy="legacy_voting",
        copula_knockoff_draws=9,
        copula_alpha_kn=0.13,
        copula_alpha_ebh=0.21,
        copula_truncation_level=2,
        enabled_methods=["copula_knockoff"],
    )
    results, all_scores = selector._copula_knockoff_selection(X, y, n_target_features=3)

    assert captured["M"] == 9
    assert np.isclose(captured["alpha_kn"], 0.13)
    assert np.isclose(captured["alpha_ebh"], 0.21)
    assert captured["truncation_level"] == 2
    assert captured["show_progress"] is False
    assert captured["fit_shape"] == X.shape
    assert captured["fit_y_shape"] == y.shape
    assert len(results["selected_indices"]) == 3
    assert results["copula_effective_truncation_level"] == {"vine_x": 2, "vine_2p": 2}
    assert len(all_scores) == X.shape[1]


def test_copula_stabilizer_repeats_runs_and_emits_metadata(monkeypatch):
    X, y = _make_binary_hdlss(79)
    call_seeds = []

    class _StubCopulaSelector:
        def __init__(self, **kwargs):
            self.seed = int(kwargs["random_state"])
            self.base_seed = 79
            self.seed_stride = 11
            call_seeds.append(self.seed)
            self.truncation_level_effective_ = {"vine_x": 3, "vine_2p": 3}

        def fit(self, X_fit, y_fit):
            return self

        def _run_id(self):
            return int((self.seed - self.base_seed) // self.seed_stride)

        def get_weights(self):
            w = np.zeros(X.shape[1], dtype=float)
            rid = self._run_id() % 3
            if rid == 0:
                w[[0, 1]] = [0.95, 0.80]
            elif rid == 1:
                w[[0, 2]] = [0.92, 0.78]
            else:
                w[[0, 1]] = [0.90, 0.76]
            return w

        def get_support(self):
            rid = self._run_id() % 3
            if rid == 0:
                return np.array([0, 1], dtype=int)
            if rid == 1:
                return np.array([0, 2], dtype=int)
            return np.array([0, 1], dtype=int)

    monkeypatch.setattr(feature_selector_module, "CopulaKnockoffSelector", _StubCopulaSelector)

    selector = FeatureSelector(
        n_bootstrap_iterations=1,
        random_state=79,
        problem_type="classification",
        selection_strategy="legacy_voting",
        copula_knockoff_draws=5,
        copula_alpha_kn=0.2,
        copula_alpha_ebh=0.3,
        copula_truncation_level=3,
        copula_stabilizer_runs=3,
        copula_stabilizer_use_ebh=True,
        copula_stabilizer_seed_stride=11,
        enabled_methods=["copula_knockoff"],
    )
    results, all_scores = selector._copula_knockoff_selection(X, y, n_target_features=3)

    assert call_seeds == [79, 90, 101]
    assert bool(results["copula_stabilizer_enabled"]) is True
    assert int(results["copula_stabilizer_runs"]) == 3
    assert bool(results["copula_stabilizer_use_ebh"]) is True
    assert int(results["copula_stabilizer_seed_stride"]) == 11
    support_freq = np.asarray(results["copula_stabilizer_support_frequency"], dtype=float)
    assert support_freq.shape[0] == X.shape[1]
    assert support_freq[0] >= support_freq[1] >= 0.0
    assert len(results["selected_indices"]) > 0
    assert len(all_scores) == X.shape[1]


def test_feature_selector_method_timeout_marks_timed_out_method(monkeypatch):
    if not hasattr(signal, "SIGALRM"):
        return

    X, y = _make_binary_hdlss(131)

    original = feature_selector_module.FeatureSelector._mutual_information_selection

    def _slow_mutual_information(self, X_data, y_data, n_target_features):
        time.sleep(0.20)
        return original(self, X_data, y_data, n_target_features)

    monkeypatch.setattr(
        feature_selector_module.FeatureSelector,
        "_mutual_information_selection",
        _slow_mutual_information,
    )

    selector = FeatureSelector(
        n_bootstrap_iterations=1,
        random_state=131,
        problem_type="classification",
        selection_strategy="mnpo_portfolio",
        method_timeout_seconds=0.05,
        enabled_methods=["mutual_information", "anova_f"],
    )

    _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
    mi_info = result.method_results.get("mutual_information", {})
    assert bool(mi_info.get("timed_out", False)) is True
    assert np.isclose(float(mi_info.get("timeout_seconds", 0.0)), 0.05)
    assert np.isclose(float(result.config.get("method_timeout_seconds", 0.0)), 0.05)


if __name__ == "__main__":
    unittest.main()

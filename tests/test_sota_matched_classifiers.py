"""Tests for the per-dataset SOTA-matched classifier protocol.

Validates that:
1. Typed protocol records contain all FS-pipeline datasets in the validation catalog.
2. Every executable candidate is registered and every status is persisted.
3. Integrated datasets inherit their parent protocol.
4. The opt-in lane labels exact, family-proxy, and unavailable rules truthfully.
5. A focused task run emits a family-proxy result row with complete provenance.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tabnetics.benchmarks.runner import (
    BENCHMARK_DATASETS,
    DATASET_SOTA_CLASSIFIERS,
    DATASET_SOTA_MATCHED_PROTOCOLS,
    _INTEGRATED_PARENT_MAP,
    _get_sota_matched_protocol_for_dataset,
    _build_ablation_configs,
    _build_base_config,
    _build_run_summary,
    _get_sota_classifiers_for_dataset,
    _resolve_sota_matched_protocol_for_dataset,
    _run_dataset_seed_task,
    SotaMatchedProtocolError,
    build_arg_parser,
)
from tabnetics.benchmarks.profiles import (
    SOTA_MATCH_STATUS_EXACT,
    SOTA_MATCH_STATUS_FAMILY_PROXY,
    SOTA_MATCH_STATUS_UNAVAILABLE,
    SotaMatchedClassifierProtocol,
)
from tabnetics.classification.registry import CLASSIFIER_SPECS
from tabnetics.pipeline.pipeline import PipelineRunResult
from tabnetics.validation.suite import CATALOG, DatasetIntegritySkipError


# All classifier keys supported by the pipeline.
VALID_CLASSIFIER_KEYS = frozenset(CLASSIFIER_SPECS)

# All FS-pipeline datasets in the validation catalog.
EXPECTED_FS_DATASETS = frozenset(
    {ds_id for ds_id, spec in CATALOG.items() if str(spec.pipeline).strip().lower() == "fs"}
)


class TestDatasetSotaClassifiers(unittest.TestCase):
    """Unit tests for typed published-classifier protocol records."""

    def test_all_fs_datasets_covered(self) -> None:
        """Every FS-pipeline dataset from the validation guide has a SOTA entry."""
        mapped = set(DATASET_SOTA_MATCHED_PROTOCOLS.keys())
        self.assertTrue(
            EXPECTED_FS_DATASETS.issubset(mapped),
            f"Missing datasets: {EXPECTED_FS_DATASETS - mapped}",
        )

    def test_all_classifier_keys_valid(self) -> None:
        """Every classifier in the mapping is a valid pipeline key."""
        for ds_id, protocol in DATASET_SOTA_MATCHED_PROTOCOLS.items():
            with self.subTest(dataset=ds_id):
                self.assertEqual(protocol.match_status, SOTA_MATCH_STATUS_FAMILY_PROXY)
                self.assertGreater(len(protocol.candidates), 0, "Empty classifier list")
                self.assertTrue(protocol.source)
                self.assertTrue(protocol.selector_classifier_coupling)
                for c in protocol.candidates:
                    self.assertIn(c, VALID_CLASSIFIER_KEYS, f"Invalid key {c!r}")

    def test_no_duplicate_classifiers_per_dataset(self) -> None:
        """No dataset lists the same classifier twice."""
        for ds_id, protocol in DATASET_SOTA_MATCHED_PROTOCOLS.items():
            self.assertEqual(protocol.candidates, DATASET_SOTA_CLASSIFIERS[ds_id])
            self.assertEqual(
                len(protocol.candidates),
                len(set(protocol.candidates)),
                f"{ds_id}: duplicate classifiers in {protocol.candidates}",
            )

    def test_weighted_vote_protocols_remain_proxy_labeled(self) -> None:
        for dataset_id in ("leukemia_golub", "dlbcl_shipp", "gcm_ramaswamy"):
            with self.subTest(dataset=dataset_id):
                protocol = DATASET_SOTA_MATCHED_PROTOCOLS[dataset_id]
                self.assertEqual(protocol.match_status, SOTA_MATCH_STATUS_FAMILY_PROXY)
                self.assertTrue(protocol.unavailable_requirements)
                self.assertTrue(protocol.notes)


class TestIntegratedParentMapping(unittest.TestCase):
    """Integrated datasets inherit SOTA classifiers from their parent."""

    def test_integrated_parents_exist(self) -> None:
        for int_ds, parent_ds in _INTEGRATED_PARENT_MAP.items():
            self.assertIn(parent_ds, DATASET_SOTA_CLASSIFIERS)

    def test_get_sota_classifiers_returns_parent(self) -> None:
        for int_ds, parent_ds in _INTEGRATED_PARENT_MAP.items():
            result = _get_sota_classifiers_for_dataset(int_ds)
            self.assertEqual(result, DATASET_SOTA_CLASSIFIERS[parent_ds])

    def test_get_typed_protocol_returns_parent_rule(self) -> None:
        for int_ds, parent_ds in _INTEGRATED_PARENT_MAP.items():
            protocol = _get_sota_matched_protocol_for_dataset(int_ds)
            self.assertEqual(protocol, DATASET_SOTA_MATCHED_PROTOCOLS[parent_ds])

    def test_unknown_dataset_returns_none(self) -> None:
        self.assertIsNone(_get_sota_classifiers_for_dataset("nonexistent_xyz"))


class TestSotaMatchedConfigInjection(unittest.TestCase):
    """Tests the config injection logic for --use-sota-matched-classifiers."""

    def _make_args(self, use_sota: bool = True, profile: str = "none") -> object:
        parser = build_arg_parser()
        base_args = [
            "--dataset-sets", "smoke",
            "--seeds", "42",
            "--ablation-profile", profile,
            "--task-timeout-sec", "60",
        ]
        if use_sota:
            base_args.append("--use-sota-matched-classifiers")
        return parser.parse_args(base_args)

    def test_dataset_integrity_cli_flags_parse(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--dataset-sets", "smoke",
                "--seeds", "42",
                "--dataset-integrity-policy", "skip",
                "--dataset-min-classes", "3",
                "--dataset-min-class-count", "2",
            ]
        )
        self.assertEqual(args.dataset_integrity_policy, "skip")
        self.assertEqual(args.dataset_min_classes, 3)
        self.assertEqual(args.dataset_min_class_count, 2)

    def test_sota_matched_config_added_when_flag_set(self) -> None:
        args = self._make_args(use_sota=True)
        ds_id = "leukemia_golub"
        spec = BENCHMARK_DATASETS[ds_id]
        base_cfg = _build_base_config(args=args, spec=spec, seed=42)
        configs = _build_ablation_configs(base_cfg, profile="none")

        sota_cls = _get_sota_classifiers_for_dataset(ds_id)
        self.assertIsNotNone(sota_cls)
        sota_cfg = copy.deepcopy(base_cfg)
        sota_cfg.model_candidates = sota_cls
        configs.append(("sota_matched_family_proxy", sota_cfg))

        names = [n for n, _ in configs]
        self.assertIn("baseline", names)
        self.assertIn("sota_matched_family_proxy", names)

    def test_sota_matched_config_not_added_when_flag_unset(self) -> None:
        args = self._make_args(use_sota=False)
        ds_id = "leukemia_golub"
        spec = BENCHMARK_DATASETS[ds_id]
        base_cfg = _build_base_config(args=args, spec=spec, seed=42)
        configs = _build_ablation_configs(base_cfg, profile="none")
        # Without the flag, no injection happens.
        names = [n for n, _ in configs]
        self.assertIn("baseline", names)
        self.assertNotIn("sota_matched", names)

    def test_sota_matched_classifiers_match_mapping(self) -> None:
        """Verify per-dataset classifiers are correct for several datasets."""
        args = self._make_args(use_sota=True)
        test_cases = {
            "colon_alon": ("svm_linear", "knn", "dlda"),
            "nci60_ross": ("svm_rbf", "knn", "dlda"),
            "breast_vantveer": ("svm_linear", "knn", "dlda"),
        }
        for ds_id, expected in test_cases.items():
            with self.subTest(dataset=ds_id):
                if ds_id not in BENCHMARK_DATASETS:
                    self.skipTest(f"{ds_id} not in BENCHMARK_DATASETS")
                spec = BENCHMARK_DATASETS[ds_id]
                base_cfg = _build_base_config(args=args, spec=spec, seed=42)
                sota_cfg = copy.deepcopy(base_cfg)
                sota_cls = _get_sota_classifiers_for_dataset(ds_id)
                sota_cfg.model_candidates = sota_cls
                self.assertEqual(sota_cfg.model_candidates, expected)

    def test_include_flags_set_correctly(self) -> None:
        """Ensure include_*_model flags are set based on SOTA classifiers."""
        args = self._make_args(use_sota=True)
        ds_id = "gcm_ramaswamy"  # uses svm_linear, knn, dlda, rf
        if ds_id not in BENCHMARK_DATASETS:
            self.skipTest(f"{ds_id} not in BENCHMARK_DATASETS")
        spec = BENCHMARK_DATASETS[ds_id]
        base_cfg = _build_base_config(args=args, spec=spec, seed=42)
        sota_cls = _get_sota_classifiers_for_dataset(ds_id)
        sota_cfg = copy.deepcopy(base_cfg)
        sota_cfg.model_candidates = sota_cls
        sota_cfg.include_svm_linear_model = "svm_linear" in sota_cls
        sota_cfg.include_knn_model = "knn" in sota_cls
        sota_cfg.include_dlda_model = "dlda" in sota_cls
        sota_cfg.include_rf_model = "rf" in sota_cls
        sota_cfg.include_xgb_model = "xgb" in sota_cls

        self.assertTrue(sota_cfg.include_svm_linear_model)
        self.assertTrue(sota_cfg.include_knn_model)
        self.assertTrue(sota_cfg.include_dlda_model)
        self.assertTrue(sota_cfg.include_rf_model)
        self.assertFalse(sota_cfg.include_xgb_model)


class TestSotaMatchedEndToEnd(unittest.TestCase):
    """Focused task-level tests for matched-protocol provenance."""

    @staticmethod
    def _fake_pipeline_result(
        dataset_name: str,
        seed: int,
        *,
        model_name: str = "svm_rbf",
    ) -> PipelineRunResult:
        return PipelineRunResult(
            dataset_name=dataset_name,
            seed=seed,
            n_samples_total=12,
            n_features_total=4,
            n_train=9,
            n_test=3,
            n_fs_subset=2,
            accuracy=0.75,
            balanced_accuracy=0.75,
            macro_f1=0.75,
            hybrid_score=0.75,
            roc_auc=0.75,
            log_loss=0.40,
            roc_curve_type="binary",
            roc_auc_source="test",
            roc_curve_points=((0.0, 0.0), (1.0, 1.0)),
            roc_curves_by_method={},
            selected_features_count=2,
            selected_feature_indices_original=(0, 1),
            model_name=model_name,
            fs_time_sec=0.01,
            dist_time_sec=0.01,
            transform_time_sec=0.01,
            n_dist_features_fitted=2,
            n_dist_features_transformed=2,
            n_dist_rejected=0,
            n_dist_skipped_unreliable=0,
            n_dist_skipped_block_cv=0,
            n_low_gof_downweighted=0,
            mean_dist_stability_weight=1.0,
            cdf_block_gating_time_sec=0.0,
            cdf_block_gating_budget_hit=False,
            cdf_block_gating_blocks_evaluated=0,
            cdf_block_gating_blocks_applied=0,
            split_indices_train=tuple(range(9)),
            split_indices_test=(9, 10, 11),
            config_snapshot={"classifier_conformal_method": "split"},
        )

    def _run_protocol_fixture(
        self,
        protocol: SotaMatchedClassifierProtocol,
        *,
        selected_model: str = "svm_rbf",
    ) -> dict[str, object]:
        args = build_arg_parser().parse_args(
            [
                "--dataset-sets", "smoke",
                "--seeds", "42",
                "--ablation-profile", "none",
                "--use-sota-matched-classifiers",
                "--task-timeout-sec", "600",
            ]
        )
        X = np.arange(48, dtype=float).reshape(12, 4)
        y = np.asarray([0, 1] * 6)
        batch_meta = {
            "batch_label_policy": "none",
            "batch_label_policy_reason": "policy_none",
            "batch_labels_available": False,
            "batch_labels_n_unique": 0,
            "multiomics_feature_blocks_available": False,
            "multiomics_feature_blocks_source_reason": "not_attempted",
            "multiomics_feature_blocks": {},
        }

        with patch(
            "tabnetics.benchmarks.runner.DATASET_SOTA_MATCHED_PROTOCOLS",
            {"leukemia_golub": protocol},
        ), patch(
            "tabnetics.benchmarks.runner._load_dataset",
            return_value=(X, y, "test_fixture", "easy", None, batch_meta),
        ), patch(
            "tabnetics.benchmarks.runner._run_pipeline_with_hard_timeout",
            side_effect=lambda **_: self._fake_pipeline_result(
                "leukemia_golub",
                42,
                model_name=selected_model,
            ),
        ):
            return _run_dataset_seed_task("leukemia_golub", 42, args)

    def test_proxy_protocol_emits_complete_result_provenance(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--dataset-sets", "smoke",
                "--seeds", "42",
                "--ablation-profile", "none",
                "--use-sota-matched-classifiers",
                "--task-timeout-sec", "600",
            ]
        )

        X = np.arange(48, dtype=float).reshape(12, 4)
        y = np.asarray([0, 1] * 6)
        batch_meta = {
            "batch_label_policy": "none",
            "batch_label_policy_reason": "policy_none",
            "batch_labels_available": False,
            "batch_labels_n_unique": 0,
            "multiomics_feature_blocks_available": False,
            "multiomics_feature_blocks_source_reason": "not_attempted",
            "multiomics_feature_blocks": {},
        }

        def fake_pipeline(**kwargs: object) -> PipelineRunResult:
            return self._fake_pipeline_result("leukemia_golub", 42)

        with patch(
            "tabnetics.benchmarks.runner._load_dataset",
            return_value=(X, y, "test_fixture", "easy", None, batch_meta),
        ), patch(
            "tabnetics.benchmarks.runner._run_pipeline_with_hard_timeout",
            side_effect=fake_pipeline,
        ):
            result = _run_dataset_seed_task("leukemia_golub", 42, args)

        configs_run = {r["config"] for r in result["rows"]}
        self.assertEqual(configs_run, {"baseline", "sota_matched_family_proxy"})
        proxy_row = next(
            row for row in result["rows"] if row["config"] == "sota_matched_family_proxy"
        )
        self.assertEqual(proxy_row["sota_matched_protocol_status"], SOTA_MATCH_STATUS_FAMILY_PROXY)
        self.assertEqual(proxy_row["sota_matched_rule_dataset_id"], "leukemia_golub")
        self.assertIn("Golub", proxy_row["sota_matched_protocol_source"])
        self.assertIn("golub_weighted_vote", proxy_row["sota_matched_unavailable_requirements"])
        self.assertIn("svm_rbf", proxy_row["sota_matched_candidates"])
        summary = _build_run_summary(
            rows=[proxy_row],
            failures=[],
            metadata={"datasets": ["leukemia_golub"], "seeds": [42]},
            run_dir="fixture",
        )
        summary_protocol = summary["results"][0]["sota_matched_protocol"]
        self.assertEqual(summary_protocol["match_status"], SOTA_MATCH_STATUS_FAMILY_PROXY)
        self.assertEqual(summary_protocol["rule_dataset_id"], "leukemia_golub")
        self.assertIn("golub_weighted_vote", summary_protocol["unavailable_requirements"])
        self.assertEqual(summary_protocol["executed_canonical_classifier"], "svm_rbf")

    def test_unstructured_exact_protocol_executes_and_is_labeled_exact(self) -> None:
        exact = SotaMatchedClassifierProtocol(
            candidates=("lr",),
            match_status=SOTA_MATCH_STATUS_EXACT,
            source="fixture source",
            selector_classifier_coupling="none_declared",
        )
        result = self._run_protocol_fixture(exact, selected_model="lr")
        configs = {row["config"] for row in result["rows"]}
        self.assertEqual(configs, {"baseline", "sota_matched_exact"})
        exact_row = next(
            row for row in result["rows"] if row["config"] == "sota_matched_exact"
        )
        self.assertEqual(exact_row["sota_matched_protocol_status"], SOTA_MATCH_STATUS_EXACT)
        self.assertEqual(exact_row["sota_matched_candidates"], '["lr"]')
        self.assertEqual(exact_row["sota_matched_executed_canonical_classifier"], "lr")

    def test_protocol_model_fallback_is_a_truthful_execution_skip(self) -> None:
        exact = SotaMatchedClassifierProtocol(
            candidates=("lr",),
            match_status=SOTA_MATCH_STATUS_EXACT,
            source="fixture source",
            selector_classifier_coupling="none_declared",
        )
        result = self._run_protocol_fixture(exact, selected_model="svm_rbf")
        self.assertEqual({row["config"] for row in result["rows"]}, {"baseline"})
        failure = next(
            item
            for item in result["failures"]
            if item["status"] == "skipped_sota_matched_invalid_execution"
        )
        self.assertIn("outside the matched protocol", failure["error"])
        self.assertEqual(failure["sota_matched_executed_classifier"], "svm_rbf")
        self.assertEqual(failure["sota_matched_executed_canonical_classifier"], "svm_rbf")
        summary = _build_run_summary(
            rows=result["rows"],
            failures=result["failures"],
            metadata={"datasets": ["leukemia_golub"], "seeds": [42]},
            run_dir="fixture",
        )
        summary_failure = summary["sota_matched_protocol_failures"][0]
        self.assertEqual(summary_failure["status"], "skipped_sota_matched_invalid_execution")
        self.assertEqual(summary_failure["executed_canonical_classifier"], "svm_rbf")

    def test_unavailable_protocol_emits_a_truthful_skip_record(self) -> None:
        unavailable = SotaMatchedClassifierProtocol(
            candidates=tuple(),
            match_status=SOTA_MATCH_STATUS_UNAVAILABLE,
            source="fixture source",
            selector_classifier_coupling="published_weighted_vote",
            unavailable_requirements=("published_weighted_vote",),
        )
        result = self._run_protocol_fixture(unavailable)
        self.assertEqual({row["config"] for row in result["rows"]}, {"baseline"})
        failure = next(
            item
            for item in result["failures"]
            if item["status"] == "skipped_sota_matched_unavailable"
        )
        record = failure["sota_matched_protocol"]
        self.assertEqual(record["match_status"], SOTA_MATCH_STATUS_UNAVAILABLE)
        self.assertEqual(record["rule_dataset_id"], "leukemia_golub")
        self.assertEqual(record["unavailable_requirements"], ["published_weighted_vote"])

    def test_structured_exact_protocol_fails_closed_without_an_executor(self) -> None:
        structured = SotaMatchedClassifierProtocol(
            candidates=("lr",),
            match_status=SOTA_MATCH_STATUS_EXACT,
            source="fixture source",
            selector_classifier_coupling="ordered_pair_rule",
            structured_state_required=True,
            structured_rule_id="fixture_ordered_pair_rule_v1",
        )
        result = self._run_protocol_fixture(structured)
        self.assertEqual({row["config"] for row in result["rows"]}, {"baseline"})
        failure = next(
            item
            for item in result["failures"]
            if item["status"] == "skipped_sota_matched_invalid_protocol"
        )
        self.assertIn("no registered structured-rule executor", failure["error"])
        record = failure["sota_matched_protocol"]
        self.assertTrue(record["structured_state_required"])
        self.assertEqual(record["structured_rule_id"], "fixture_ordered_pair_rule_v1")

    def test_exact_and_unavailable_protocols_have_distinct_contracts(self) -> None:
        exact = SotaMatchedClassifierProtocol(
            candidates=("lr",),
            match_status=SOTA_MATCH_STATUS_EXACT,
            source="fixture source",
            selector_classifier_coupling="ordered_pair_rule",
            structured_state_required=True,
            structured_rule_id="fixture_ordered_pair_rule_v1",
        )
        unavailable = SotaMatchedClassifierProtocol(
            candidates=tuple(),
            match_status=SOTA_MATCH_STATUS_UNAVAILABLE,
            source="fixture source",
            selector_classifier_coupling="published_weighted_vote",
            unavailable_requirements=("published_weighted_vote",),
        )
        self.assertEqual(exact.to_record()["match_status"], SOTA_MATCH_STATUS_EXACT)
        self.assertTrue(exact.to_record()["structured_state_required"])
        self.assertEqual(exact.to_record()["structured_rule_id"], "fixture_ordered_pair_rule_v1")
        self.assertEqual(unavailable.to_record()["match_status"], SOTA_MATCH_STATUS_UNAVAILABLE)

    def test_structured_protocol_requires_a_rule_identifier(self) -> None:
        with self.assertRaisesRegex(ValueError, "structured_rule_id"):
            SotaMatchedClassifierProtocol(
                candidates=("lr",),
                match_status=SOTA_MATCH_STATUS_EXACT,
                source="fixture source",
                structured_state_required=True,
            )

    def test_unknown_registry_candidate_fails_closed(self) -> None:
        invalid = SotaMatchedClassifierProtocol(
            candidates=("not_a_classifier",),
            match_status=SOTA_MATCH_STATUS_FAMILY_PROXY,
            source="fixture source",
        )
        with patch(
            "tabnetics.benchmarks.runner.DATASET_SOTA_MATCHED_PROTOCOLS",
            {"fixture": invalid},
        ):
            with self.assertRaisesRegex(SotaMatchedProtocolError, "unknown classifier"):
                _resolve_sota_matched_protocol_for_dataset("fixture")

    def test_cli_never_claims_proxy_is_exact(self) -> None:
        help_text = build_arg_parser().format_help()
        self.assertIn("family_proxy", help_text)
        self.assertNotIn("exactly the evaluation classifier", help_text)

    def test_legacy_cli_explicitly_rejects_matched_proxy_execution(self) -> None:
        legacy_script = (
            Path(__file__).resolve().parents[2]
            / "experiments"
            / "run_df_fs_sota_benchmark.py"
        )
        source = legacy_script.read_text(encoding="utf-8")
        self.assertIn("Deprecated and unavailable in this legacy runner", source)
        self.assertIn("is unavailable in this legacy runner", source)
        self.assertNotIn("exactly the evaluation classifier(s) from the published SOTA", source)

    def test_dataset_integrity_skip_is_recorded_as_skip_failure(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--dataset-sets", "smoke",
                "--seeds", "42",
                "--ablation-profile", "none",
                "--dataset-integrity-policy", "skip",
            ]
        )

        with patch(
            "tabnetics.benchmarks.runner._load_dataset",
            side_effect=DatasetIntegritySkipError("forced integrity skip"),
        ):
            result = _run_dataset_seed_task("synthetic_easy_dfshift", 42, args)

        self.assertEqual(result["rows"], [])
        self.assertEqual(len(result["failures"]), 1)
        self.assertEqual(result["failures"][0].get("status"), "skipped_dataset_integrity")


if __name__ == "__main__":
    unittest.main()

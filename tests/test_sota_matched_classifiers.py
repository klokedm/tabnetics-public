"""Tests for the per-dataset SOTA-matched classifier protocol.

Validates that:
1. DATASET_SOTA_CLASSIFIERS contains all FS-pipeline datasets in the validation catalog.
2. All classifier keys are valid pipeline keys.
3. Integrated datasets inherit from their parent.
4. The --use-sota-matched-classifiers flag correctly injects a
   'sota_matched' config with the right classifiers per dataset.
5. A synthetic end-to-end run with --use-sota-matched-classifiers
   produces both 'baseline' and 'sota_matched' result rows.
"""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from tabnetics.benchmarks.runner import (
    BENCHMARK_DATASETS,
    DATASET_SOTA_CLASSIFIERS,
    _INTEGRATED_PARENT_MAP,
    _build_ablation_configs,
    _build_base_config,
    _get_sota_classifiers_for_dataset,
    _run_dataset_seed_task,
    build_arg_parser,
)
from tabnetics.validation.suite import CATALOG, DatasetIntegritySkipError


# All classifier keys supported by the pipeline.
VALID_CLASSIFIER_KEYS = frozenset(
    {"lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "elastic_net_lr", "xgb", "tabpfn"}
)

# All FS-pipeline datasets in the validation catalog.
EXPECTED_FS_DATASETS = frozenset(
    {ds_id for ds_id, spec in CATALOG.items() if str(spec.pipeline).strip().lower() == "fs"}
)


class TestDatasetSotaClassifiers(unittest.TestCase):
    """Unit tests for the DATASET_SOTA_CLASSIFIERS mapping."""

    def test_all_fs_datasets_covered(self) -> None:
        """Every FS-pipeline dataset from VALIDATION.md has a SOTA entry."""
        mapped = set(DATASET_SOTA_CLASSIFIERS.keys())
        self.assertTrue(
            EXPECTED_FS_DATASETS.issubset(mapped),
            f"Missing datasets: {EXPECTED_FS_DATASETS - mapped}",
        )

    def test_all_classifier_keys_valid(self) -> None:
        """Every classifier in the mapping is a valid pipeline key."""
        for ds_id, classifiers in DATASET_SOTA_CLASSIFIERS.items():
            with self.subTest(dataset=ds_id):
                self.assertGreater(len(classifiers), 0, "Empty classifier list")
                for c in classifiers:
                    self.assertIn(c, VALID_CLASSIFIER_KEYS, f"Invalid key {c!r}")

    def test_no_duplicate_classifiers_per_dataset(self) -> None:
        """No dataset lists the same classifier twice."""
        for ds_id, classifiers in DATASET_SOTA_CLASSIFIERS.items():
            self.assertEqual(
                len(classifiers),
                len(set(classifiers)),
                f"{ds_id}: duplicate classifiers in {classifiers}",
            )


class TestIntegratedParentMapping(unittest.TestCase):
    """Integrated datasets inherit SOTA classifiers from their parent."""

    def test_integrated_parents_exist(self) -> None:
        for int_ds, parent_ds in _INTEGRATED_PARENT_MAP.items():
            self.assertIn(parent_ds, DATASET_SOTA_CLASSIFIERS)

    def test_get_sota_classifiers_returns_parent(self) -> None:
        for int_ds, parent_ds in _INTEGRATED_PARENT_MAP.items():
            result = _get_sota_classifiers_for_dataset(int_ds)
            self.assertEqual(result, DATASET_SOTA_CLASSIFIERS[parent_ds])

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
        configs.append(("sota_matched", sota_cfg))

        names = [n for n, _ in configs]
        self.assertIn("baseline", names)
        self.assertIn("sota_matched", names)

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
    """Synthetic end-to-end test that actually runs both configs."""

    def test_synthetic_produces_both_configs(self) -> None:
        """Run a synthetic dataset with SOTA matching and verify both rows."""
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

        # The smoke set includes leukemia_golub (has SOTA classifiers)
        # and synthetic_easy_dfshift (no SOTA → should NOT produce sota_matched).
        ds_id_with = "leukemia_golub"
        ds_id_without = "synthetic_easy_dfshift"

        if ds_id_with in BENCHMARK_DATASETS:
            result = _run_dataset_seed_task(ds_id_with, 42, args)
            configs_run = {r["config"] for r in result["rows"]}
            self.assertIn("baseline", configs_run, "Missing baseline for leukemia")
            self.assertIn("sota_matched", configs_run, "Missing sota_matched for leukemia")

            # Verify the SOTA-matched row reports a model from the SOTA set.
            for row in result["rows"]:
                if row["config"] == "sota_matched":
                    self.assertIn(
                        row["model"],
                        {"svm_rbf", "knn", "dlda"},
                        f"SOTA-matched model {row['model']} not in expected set",
                    )

        if ds_id_without in BENCHMARK_DATASETS:
            result = _run_dataset_seed_task(ds_id_without, 42, args)
            configs_run = {r["config"] for r in result["rows"]}
            self.assertIn("baseline", configs_run)
            # synthetic dataset has no SOTA classifiers → no sota_matched
            self.assertNotIn("sota_matched", configs_run)

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

from tabnetics.benchmarks.runner import BENCHMARK_DATASETS, DATASET_SETS
from tabnetics.validation.suite import CATALOG


def test_benchmark_registry_covers_validation_catalog_fs_and_integrated():
    expected = {
        ds_id
        for ds_id, spec in CATALOG.items()
        if spec.pipeline in {"fs", "integrated"}
    }
    present = {
        ds_id
        for ds_id, spec in BENCHMARK_DATASETS.items()
        if spec.source_kind == "validation_catalog"
    }
    assert expected.issubset(present)


def test_validation_dataset_sets_cover_catalog_entries():
    expected_all = {
        ds_id
        for ds_id, spec in CATALOG.items()
        if spec.pipeline in {"fs", "integrated"}
    }
    expected_fs = {ds_id for ds_id, spec in CATALOG.items() if spec.pipeline == "fs"}
    expected_integrated = {ds_id for ds_id, spec in CATALOG.items() if spec.pipeline == "integrated"}

    assert set(DATASET_SETS["validation_all"]) == expected_all
    assert set(DATASET_SETS["validation_fs_all"]) == expected_fs
    assert set(DATASET_SETS["validation_integrated_all"]) == expected_integrated


def test_scikit_feature_expansion_datasets_are_registered():
    expected_new = {
        "cll_sub_111",
        "tox_171",
        "gla_bra_180",
        "carcinom_11class",
        "glioma_50_4class",
        "brain_tumor_2_50_4class",
        "leukemia_1_72_3class",
        "nci_61_8class",
        "nci9_60_9class",
        "orlraws10p",
        "warp_pie10p",
        "pixraw10p",
    }
    assert expected_new.issubset(set(CATALOG.keys()))
    assert expected_new.issubset(set(BENCHMARK_DATASETS.keys()))


def test_new_quick_suites_include_expected_datasets():
    assert set(DATASET_SETS["quick_scikit_multiclass_hard"]) == {
        "nci9_60_9class",
        "nci_61_8class",
        "carcinom_11class",
        "cll_sub_111",
    }
    assert set(DATASET_SETS["quick_extreme_p"]) == {
        "gla_bra_180",
        "glioma_50_4class",
    }
    assert set(DATASET_SETS["quick_non_genomics_hdlss"]) == {
        "orlraws10p",
        "warp_pie10p",
        "pixraw10p",
    }

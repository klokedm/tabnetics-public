from __future__ import annotations


def test_all_fs_and_integrated_catalog_datasets_have_resolvable_sota_ranges() -> None:
    # SOTA ranges are consumed by reporting and SOTA-comparison CSVs; ensure the registry
    # stays in sync with the validation catalog.
    from tabnetics.benchmarks.runner import BENCHMARK_DATASETS
    from tabnetics.validation.suite import CATALOG

    missing = []
    for ds_id, spec in CATALOG.items():
        if spec.pipeline not in {"fs", "integrated"}:
            continue
        bench = BENCHMARK_DATASETS.get(ds_id)
        if bench is None:
            missing.append(ds_id)
            continue

        lo, hi = bench.sota_holdout_bal_acc
        assert 0.0 <= float(lo) <= float(hi) <= 1.0

        ilo, ihi = bench.sota_inflated_bal_acc
        assert 0.0 <= float(ilo) <= float(ihi) <= 1.0

    assert not missing, f"Missing fs/integrated datasets in BENCHMARK_DATASETS: {missing}"


def test_integrated_datasets_inherit_parent_sota_ranges_when_available() -> None:
    from tabnetics.benchmarks.runner import BENCHMARK_DATASETS
    from tabnetics.validation.suite import CATALOG

    for ds_id, spec in CATALOG.items():
        if spec.pipeline != "integrated":
            continue
        base = str(spec.params.get("base_dataset", "") or "").strip()
        assert base, f"Integrated dataset {ds_id} is missing params.base_dataset"
        assert base in BENCHMARK_DATASETS, f"Base dataset {base} not found for integrated dataset {ds_id}"

        child = BENCHMARK_DATASETS[ds_id]
        parent = BENCHMARK_DATASETS[base]
        assert child.sota_holdout_bal_acc == parent.sota_holdout_bal_acc
        assert child.sota_inflated_bal_acc == parent.sota_inflated_bal_acc


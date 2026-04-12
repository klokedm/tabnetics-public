from __future__ import annotations


def test_dataset_registry_keys_match_dataset_id() -> None:
    from tabnetics.datasets.registry import DATASET_REGISTRY

    mismatched = []
    for key, spec in DATASET_REGISTRY.items():
        if str(getattr(spec, "dataset_id", "")) != str(key):
            mismatched.append((str(key), str(getattr(spec, "dataset_id", ""))))
    assert not mismatched, f"DatasetSpec.dataset_id mismatch (showing up to 5): {mismatched[:5]}"


def test_registry_fs_and_integrated_entries_have_sota_and_provenance() -> None:
    from tabnetics.datasets.registry import DATASET_REGISTRY

    missing = []
    for ds_id, spec in DATASET_REGISTRY.items():
        pipeline = str(getattr(spec, "pipeline", "")).strip().lower()
        if pipeline not in {"fs", "integrated"}:
            continue
        source_kind = str(getattr(spec, "source_kind", "")).strip().lower()
        if source_kind not in {"validation_catalog", "synthetic"}:
            continue

        prov = str(getattr(spec, "provenance", "") or "").strip()
        if not prov:
            missing.append((str(ds_id), "provenance"))

        strict = getattr(spec, "sota_holdout_bal_acc", None)
        inflated = getattr(spec, "sota_inflated_bal_acc", None)
        if strict is None:
            missing.append((str(ds_id), "sota_holdout_bal_acc"))
        else:
            lo, hi = strict
            assert 0.0 <= float(lo) <= float(hi) <= 1.0
        if inflated is None:
            missing.append((str(ds_id), "sota_inflated_bal_acc"))
        else:
            lo, hi = inflated
            assert 0.0 <= float(lo) <= float(hi) <= 1.0

    assert not missing, f"Missing required registry fields (showing up to 10): {missing[:10]}"


def test_validation_suite_catalog_is_view_of_registry() -> None:
    from tabnetics.datasets.registry import DATASET_REGISTRY
    from tabnetics.datasets.validation_catalog import CATALOG

    # Synthetic benchmark datasets must not appear in the validation suite catalog.
    assert "synthetic_easy_dfshift" not in CATALOG
    assert "synthetic_medium_mixed" not in CATALOG
    assert "synthetic_very_hard_sparse" not in CATALOG

    for ds_id, spec in CATALOG.items():
        assert ds_id in DATASET_REGISTRY
        # Catalog entries should reference the exact same objects from the registry.
        assert spec is DATASET_REGISTRY[ds_id]

    for ds_id, spec in DATASET_REGISTRY.items():
        if str(getattr(spec, "source_kind", "")).strip().lower() == "synthetic":
            continue
        assert ds_id in CATALOG


def test_benchmark_registry_matches_dataset_registry_core_fields() -> None:
    from tabnetics.datasets.registry import DATASET_REGISTRY
    from tabnetics.datasets.benchmark_catalog import BENCHMARK_DATASETS

    for ds_id, bench in BENCHMARK_DATASETS.items():
        reg = DATASET_REGISTRY[ds_id]
        assert str(bench.tier) == str(getattr(reg, "tier", ""))
        assert str(bench.source_kind) == str(getattr(reg, "source_kind", ""))
        assert str(bench.domain) == str(getattr(reg, "domain", ""))
        assert str(bench.platform) == str(getattr(reg, "platform", ""))
        assert bench.sota_holdout_bal_acc == getattr(reg, "sota_holdout_bal_acc")
        assert bench.sota_inflated_bal_acc == getattr(reg, "sota_inflated_bal_acc")

        if str(bench.source_kind) == "validation_catalog":
            assert bench.validation_dataset_id == ds_id
            assert bench.validation_pipeline == str(getattr(reg, "pipeline", ""))
        else:
            assert bench.validation_dataset_id is None
            assert bench.validation_pipeline is None


def test_domain_relabel_nips03_and_face_to_non_genomic() -> None:
    from tabnetics.datasets.registry import DATASET_REGISTRY

    for ds_id in ("arcene_nips03", "madelon_nips03", "gisette_nips03", "dexter_nips03", "dorothea_nips03"):
        assert str(DATASET_REGISTRY[ds_id].domain) == "non_genomic"

    for ds_id in ("orlraws10p", "warp_pie10p", "pixraw10p"):
        assert str(DATASET_REGISTRY[ds_id].domain) == "non_genomic"


def test_platform_metadata_mapping() -> None:
    from tabnetics.datasets.registry import DATASET_REGISTRY

    assert str(DATASET_REGISTRY["ovarian_petricoin"].platform) == "mass-spec"
    assert str(DATASET_REGISTRY["leukemia_golub"].platform) == "Affy HG-U95"
    assert str(DATASET_REGISTRY["colon_alon"].platform) == "Affy HG-U133A"
    assert str(DATASET_REGISTRY["synthetic_easy_dfshift"].platform) == "synthetic"

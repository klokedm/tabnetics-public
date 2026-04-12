from __future__ import annotations


def test_dataset_registry_domains_mark_proxies_and_synthetics() -> None:
    from tabnetics.datasets.registry import DATASET_REGISTRY

    assert DATASET_REGISTRY["leukemia_golub"].domain == "genomics"
    assert DATASET_REGISTRY["orlraws10p"].domain == "non_genomic"
    assert DATASET_REGISTRY["synthetic_easy_dfshift"].domain == "synthetic"



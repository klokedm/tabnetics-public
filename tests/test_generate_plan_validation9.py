from collections import Counter, defaultdict

from tabnetics.datasets.benchmark_catalog import DATASET_SETS
from tabnetics.validation.generate_plan import (
    VALIDATION_SEEDS,
    _balanced_shard_assign_validation9_pairs,
    build_jobs_validation9,
)


def _profile_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("/")
    assert len(parts) == 3
    return parts[1]


def test_build_jobs_validation9_has_expected_job_topology(tmp_path):
    jobs = build_jobs_validation9(dataset_shards=6, val8_root=tmp_path)
    assert len(jobs) == 12  # 2 profiles x 6 dataset shards

    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert profile_counts == {
        "legacy_full": 6,
        "mnpo_hybrid": 6,
    }

    for job in jobs:
        assert job.kind == "run_df_fs_sota_benchmark"
        assert list(job.params.get("seeds") or []) == list(VALIDATION_SEEDS)


def test_build_jobs_validation9_uses_67_dataset_catalog_without_rv(tmp_path):
    jobs = build_jobs_validation9(dataset_shards=6, val8_root=tmp_path)
    expected = {
        ds for ds in DATASET_SETS.get("validation_all", []) if not str(ds).startswith("rv_")
    }
    assert len(expected) == 67

    by_profile = defaultdict(set)
    for job in jobs:
        profile = _profile_from_job_id(job.job_id)
        datasets = {str(ds) for ds in (job.params.get("datasets") or [])}
        assert not any(ds.startswith("rv_") for ds in datasets)
        by_profile[profile].update(datasets)

    for profile in ("legacy_full", "mnpo_hybrid"):
        assert by_profile[profile] == expected


def test_build_jobs_validation9_profiles_encode_expected_flags(tmp_path):
    jobs = build_jobs_validation9(dataset_shards=6, val8_root=tmp_path)
    by_profile = defaultdict(list)
    for job in jobs:
        by_profile[_profile_from_job_id(job.job_id)].append(job)

    def _args(profile: str):
        return list(by_profile[profile][0].params.get("extra_args") or [])

    legacy_args = _args("legacy_full")
    assert "--classifier-selection-mode" in legacy_args
    assert legacy_args[legacy_args.index("--classifier-selection-mode") + 1] == "legacy"
    assert "--dist-criterion" in legacy_args
    assert legacy_args[legacy_args.index("--dist-criterion") + 1] == "mnpo_oracle"
    assert "--df-mnpo-include-crps" in legacy_args
    assert "--df-mnpo-include-preq" in legacy_args
    assert "--enable-stage2-ratio-augmentation" in legacy_args
    assert "--enable-maqc-pairing" in legacy_args
    assert "--enable-fs-runtime-racing" in legacy_args
    assert "--enable-diversity-oracle" in legacy_args
    assert "--fs-use-interaction-oracle" in legacy_args
    assert "--fs-use-conformal-uq" in legacy_args
    assert "--fs-compute-tremble-sensitivity" in legacy_args

    hybrid_args = _args("mnpo_hybrid")
    assert "--classifier-selection-mode" in hybrid_args
    assert hybrid_args[hybrid_args.index("--classifier-selection-mode") + 1] == "mnpo_hybrid"
    assert "--classifier-oracle-k" in hybrid_args
    assert hybrid_args[hybrid_args.index("--classifier-oracle-k") + 1] == "2"
    assert "--enable-classifier-oracle-ensemble" in hybrid_args
    assert "--enable-stage2-ratio-augmentation" in hybrid_args
    assert "--fs-use-conformal-uq" in hybrid_args

    assert "--model-candidates" in hybrid_args
    cidx = hybrid_args.index("--model-candidates") + 1
    candidates = []
    while cidx < len(hybrid_args) and not str(hybrid_args[cidx]).startswith("--"):
        candidates.append(str(hybrid_args[cidx]))
        cidx += 1
    for required in ("nsc", "pls_da_classifier", "gpc", "lgbm", "extra_tree", "catboost"):
        assert required in candidates


def test_validation9_paired_sharding_keeps_profiles_together(tmp_path):
    jobs = build_jobs_validation9(dataset_shards=6, val8_root=tmp_path)
    shards = _balanced_shard_assign_validation9_pairs(jobs, num_shards=6)
    assert set(shards.keys()) == {1, 2, 3, 4, 5, 6}
    for _, job_ids in shards.items():
        # Every shard should get one dataset-partition pair = 2 jobs.
        assert len(job_ids) == 2
        profiles = {_profile_from_job_id(jid) for jid in job_ids}
        assert profiles == {"legacy_full", "mnpo_hybrid"}

from collections import Counter, defaultdict

from tabnetics.datasets.benchmark_catalog import DATASET_SETS
from tabnetics.validation.generate_plan import (
    VALIDATION_SEEDS,
    _balanced_shard_assign_validation6_pairs,
    build_jobs_validation6,
)


def _profile_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("/")
    assert len(parts) == 3
    return parts[1]


def test_build_jobs_validation6_has_expected_job_topology(tmp_path):
    jobs = build_jobs_validation6(dataset_shards=6, val5_root=tmp_path)
    assert len(jobs) == 12  # 2 profiles x 6 dataset shards

    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert profile_counts == {
        "baseline": 6,
        "full": 6,
    }

    for job in jobs:
        assert job.kind == "run_df_fs_sota_benchmark"
        assert list(job.params.get("seeds") or []) == list(VALIDATION_SEEDS)


def test_build_jobs_validation6_uses_67_dataset_catalog_without_rv(tmp_path):
    jobs = build_jobs_validation6(dataset_shards=6, val5_root=tmp_path)
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

    for profile in ("baseline", "full"):
        assert by_profile[profile] == expected


def test_build_jobs_validation6_profiles_encode_expected_flags(tmp_path):
    jobs = build_jobs_validation6(dataset_shards=6, val5_root=tmp_path)
    by_profile = defaultdict(list)
    for job in jobs:
        by_profile[_profile_from_job_id(job.job_id)].append(job)

    def _args(profile: str):
        return list(by_profile[profile][0].params.get("extra_args") or [])

    # Baseline: production default with PLS-DA
    baseline_args = _args("baseline")
    assert "--folding-method" in baseline_args
    assert baseline_args[baseline_args.index("--folding-method") + 1] == "pls_da"

    # Full: everything enabled
    full_args = _args("full")
    assert "--fs-use-cvar-oracle" in full_args
    assert "--fs-diversity-oracle-mode" in full_args
    assert full_args[full_args.index("--fs-diversity-oracle-mode") + 1] == "complementarity"
    assert "--fs-oracle-weighting-mode" in full_args
    assert full_args[full_args.index("--fs-oracle-weighting-mode") + 1] == "shapley"
    assert "--fs-use-ubayfs-oracle" in full_args
    assert "--enable-fs-adaptive-portfolio-sizing" in full_args
    assert "--fs-adaptive-size-min" in full_args
    assert full_args[full_args.index("--fs-adaptive-size-min") + 1] == "4"
    assert "--fs-adaptive-size-max" in full_args
    assert full_args[full_args.index("--fs-adaptive-size-max") + 1] == "8"
    assert "--fs-portfolio-size-guard" in full_args
    assert full_args[full_args.index("--fs-portfolio-size-guard") + 1] == "warn"
    # Full profile includes PLS-DA
    assert "--folding-method" in full_args
    assert full_args[full_args.index("--folding-method") + 1] == "pls_da"
    # Full profile includes WSNR prefilter
    assert "--prefilter-wsnr-enabled" in full_args
    assert "--prefilter-strategies" in full_args
    strategies = full_args[full_args.index("--prefilter-strategies") + 1]
    assert "wsnr" in strategies
    assert "mi_ftest_blend" in strategies


def test_validation6_paired_sharding_keeps_profiles_together(tmp_path):
    jobs = build_jobs_validation6(dataset_shards=6, val5_root=tmp_path)
    shards = _balanced_shard_assign_validation6_pairs(jobs, num_shards=6)
    assert set(shards.keys()) == {1, 2, 3, 4, 5, 6}
    for _, job_ids in shards.items():
        # Every shard should get one dataset-partition pair = 2 jobs.
        assert len(job_ids) == 2
        profiles = {_profile_from_job_id(jid) for jid in job_ids}
        assert profiles == {"baseline", "full"}

from collections import Counter, defaultdict

from tabnetics.datasets.benchmark_catalog import DATASET_SETS
from tabnetics.validation.generate_plan import (
    VALIDATION_SEEDS,
    _balanced_shard_assign_validation8_pairs,
    build_jobs_validation8,
)


def _profile_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("/")
    assert len(parts) == 3
    return parts[1]


def test_build_jobs_validation8_has_expected_job_topology(tmp_path):
    jobs = build_jobs_validation8(dataset_shards=6, val7_root=tmp_path)
    assert len(jobs) == 12  # 2 profiles x 6 dataset shards

    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert profile_counts == {
        "baseline": 6,
        "candidate": 6,
    }

    for job in jobs:
        assert job.kind == "run_df_fs_sota_benchmark"
        assert list(job.params.get("seeds") or []) == list(VALIDATION_SEEDS)


def test_build_jobs_validation8_uses_67_dataset_catalog_without_rv(tmp_path):
    jobs = build_jobs_validation8(dataset_shards=6, val7_root=tmp_path)
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

    for profile in ("baseline", "candidate"):
        assert by_profile[profile] == expected


def test_build_jobs_validation8_profiles_encode_expected_flags(tmp_path):
    jobs = build_jobs_validation8(dataset_shards=6, val7_root=tmp_path)
    by_profile = defaultdict(list)
    for job in jobs:
        by_profile[_profile_from_job_id(job.job_id)].append(job)

    def _args(profile: str):
        return list(by_profile[profile][0].params.get("extra_args") or [])

    baseline_args = _args("baseline")
    assert "--folding-method" in baseline_args
    assert baseline_args[baseline_args.index("--folding-method") + 1] == "pls_da"
    assert "--enable-prefilter-rnaseq-nb-lrt" not in baseline_args
    assert "--classification-backend" not in baseline_args

    candidate_args = _args("candidate")
    # Carry-over from Val-7 candidate.
    assert "--shapley-bayesian-shrinkage" in candidate_args
    assert "--adaptive-sizing-variance-penalty" in candidate_args
    # New RNA-seq prefilter signal in Val-8.
    assert "--enable-prefilter-rnaseq-nb-lrt" in candidate_args
    assert "--prefilter-rnaseq-nb-lrt-alpha" in candidate_args
    assert candidate_args[candidate_args.index("--prefilter-rnaseq-nb-lrt-alpha") + 1] == "0.10"
    # New Stage-2 classifier backend support in Val-8.
    assert "--classification-backend" in candidate_args
    assert candidate_args[candidate_args.index("--classification-backend") + 1] == "flaml"
    assert "--flaml-time-budget" in candidate_args
    assert candidate_args[candidate_args.index("--flaml-time-budget") + 1] == "90"

    assert "--model-candidates" in candidate_args
    cidx = candidate_args.index("--model-candidates") + 1
    candidates = []
    while cidx < len(candidate_args) and not str(candidate_args[cidx]).startswith("--"):
        candidates.append(str(candidate_args[cidx]))
        cidx += 1
    assert "shrinkage_lda" in candidates
    assert "nsc" in candidates
    assert "pls_da_classifier" in candidates
    assert "gpc" in candidates

    assert "--enable-model-cv-runtime-containment" in candidate_args
    assert "--model-cv-runtime-max-candidates" in candidate_args
    assert candidate_args[candidate_args.index("--model-cv-runtime-max-candidates") + 1] == "8"


def test_validation8_paired_sharding_keeps_profiles_together(tmp_path):
    jobs = build_jobs_validation8(dataset_shards=6, val7_root=tmp_path)
    shards = _balanced_shard_assign_validation8_pairs(jobs, num_shards=6)
    assert set(shards.keys()) == {1, 2, 3, 4, 5, 6}
    for _, job_ids in shards.items():
        # Every shard should get one dataset-partition pair = 2 jobs.
        assert len(job_ids) == 2
        profiles = {_profile_from_job_id(jid) for jid in job_ids}
        assert profiles == {"baseline", "candidate"}

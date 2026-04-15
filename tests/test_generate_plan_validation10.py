from collections import Counter, defaultdict

from tabnetics.datasets.benchmark_catalog import DATASET_SETS
from tabnetics.validation.generate_plan import (
    VALIDATION_SEEDS,
    _load_hf_manifest_metadata,
    _balanced_shard_assign_validation10_pairs,
    build_jobs_validation10,
)


def _profile_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("/")
    assert len(parts) == 3
    return parts[1]


def test_build_jobs_validation10_has_expected_job_topology(tmp_path):
    jobs = build_jobs_validation10(dataset_shards=6, val9_root=tmp_path)
    assert len(jobs) == 12  # 2 profiles x 6 dataset shards

    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert profile_counts == {
        "simple_all_stages": 6,
        "mnpo_all_stages": 6,
    }

    for job in jobs:
        assert job.kind == "run_df_fs_sota_benchmark"
        assert list(job.params.get("seeds") or []) == list(VALIDATION_SEEDS)


def test_build_jobs_validation10_uses_extended_validation_catalog(tmp_path):
    jobs = build_jobs_validation10(dataset_shards=6, val9_root=tmp_path)
    validation_all = {str(ds) for ds in DATASET_SETS.get("validation_all", [])}
    hf_ids = {str(ds) for ds in _load_hf_manifest_metadata().keys()}
    expected = validation_all & hf_ids
    assert len(expected) >= 68
    assert any(str(ds).startswith("rv_") for ds in expected)

    by_profile = defaultdict(set)
    for job in jobs:
        profile = _profile_from_job_id(job.job_id)
        datasets = {str(ds) for ds in (job.params.get("datasets") or [])}
        by_profile[profile].update(datasets)

    for profile in ("simple_all_stages", "mnpo_all_stages"):
        assert by_profile[profile] == expected


def test_build_jobs_validation10_profiles_encode_stage_modes(tmp_path):
    jobs = build_jobs_validation10(dataset_shards=6, val9_root=tmp_path)
    by_profile = defaultdict(list)
    for job in jobs:
        by_profile[_profile_from_job_id(job.job_id)].append(job)

    def _args(profile: str):
        return list(by_profile[profile][0].params.get("extra_args") or [])

    simple_args = _args("simple_all_stages")
    assert "--dist-criterion" in simple_args
    assert simple_args[simple_args.index("--dist-criterion") + 1] == "simple"
    assert "--classifier-selection-mode" in simple_args
    assert simple_args[simple_args.index("--classifier-selection-mode") + 1] == "legacy"
    assert "--classification-backend" in simple_args
    assert simple_args[simple_args.index("--classification-backend") + 1] == "sklearn"

    mnpo_args = _args("mnpo_all_stages")
    assert "--dist-criterion" in mnpo_args
    assert mnpo_args[mnpo_args.index("--dist-criterion") + 1] == "mnpo_oracle"
    assert "--classifier-selection-mode" in mnpo_args
    assert mnpo_args[mnpo_args.index("--classifier-selection-mode") + 1] == "mnpo_hybrid"
    assert "--enable-fs-rashomon" in mnpo_args
    assert "--df-mnpo-include-preq" in mnpo_args
    assert "--fs-use-conformal-uq" in mnpo_args


def test_validation10_paired_sharding_keeps_profiles_together(tmp_path):
    jobs = build_jobs_validation10(dataset_shards=6, val9_root=tmp_path)
    shards = _balanced_shard_assign_validation10_pairs(jobs, num_shards=6)
    assert set(shards.keys()) == {1, 2, 3, 4, 5, 6}
    for _, job_ids in shards.items():
        assert len(job_ids) == 2
        profiles = {_profile_from_job_id(jid) for jid in job_ids}
        assert profiles == {"simple_all_stages", "mnpo_all_stages"}

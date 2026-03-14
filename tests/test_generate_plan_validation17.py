from collections import Counter
from pathlib import Path

from tabnetics.validation.generate_plan import (
    VALIDATION17_DATASETS,
    VALIDATION17_PROFILE_MANIFEST,
    _balanced_shard_assign_validation17_bundles,
    build_jobs_validation17,
)


def _profile_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("/")
    assert len(parts) == 3
    return parts[1]


def _part_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("/")
    assert len(parts) == 3
    return parts[2]


def test_build_jobs_validation17_has_expected_topology(tmp_path):
    jobs = build_jobs_validation17(dataset_shards=3, val15_root=Path(tmp_path))
    assert len(jobs) == 11 * 3

    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert set(profile_counts.keys()) == set(VALIDATION17_PROFILE_MANIFEST.keys())
    assert all(v == 3 for v in profile_counts.values())

    datasets = {
        ds_id
        for job in jobs
        for ds_id in list(job.params.get("datasets") or [])
    }
    assert datasets == set(VALIDATION17_DATASETS)


def test_validation17_profile_flags_and_guards_are_wired(tmp_path):
    jobs = build_jobs_validation17(dataset_shards=2, val15_root=Path(tmp_path))
    by_profile = {}
    for job in jobs:
        by_profile.setdefault(_profile_from_job_id(job.job_id), []).append(job)

    for job in jobs:
        assert str(job.params.get("dataset_integrity_policy")) == "error"
        assert bool(job.params.get("allow_synthetic_fallback", True)) is False
        extra_args = list(job.params.get("extra_args") or [])
        assert extra_args[extra_args.index("--df-stage-position") + 1] == "after_fs"

    ref_job = by_profile["v16_ref"][0]
    assert str(ref_job.params.get("fs_method_set")) == "mnpo_v14_core_plus_ipss"


def test_validation17_bundle_sharding_keeps_profiles_together(tmp_path):
    jobs = build_jobs_validation17(dataset_shards=4, val15_root=Path(tmp_path))
    shards = _balanced_shard_assign_validation17_bundles(jobs, num_shards=4)

    assert set(shards.keys()) == {1, 2, 3, 4}
    for _, job_ids in shards.items():
        assert len(job_ids) == 11
        profiles = {_profile_from_job_id(jid) for jid in job_ids}
        assert profiles == set(VALIDATION17_PROFILE_MANIFEST.keys())
        parts = {_part_from_job_id(jid) for jid in job_ids}
        assert len(parts) == 1

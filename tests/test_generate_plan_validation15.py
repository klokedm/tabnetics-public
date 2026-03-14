from collections import Counter
from pathlib import Path

from tabnetics.validation.generate_plan import (
    VALIDATION15_PROFILE_MANIFEST,
    _balanced_shard_assign_validation15_bundles,
    build_jobs_validation15,
)


def _profile_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("/")
    assert len(parts) == 3
    return parts[1]


def _part_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("/")
    assert len(parts) == 3
    return parts[2]


def test_build_jobs_validation15_has_expected_topology(tmp_path):
    jobs = build_jobs_validation15(dataset_shards=3, val14_root=Path(tmp_path))
    assert len(jobs) == 9 * 3

    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert set(profile_counts.keys()) == set(VALIDATION15_PROFILE_MANIFEST.keys())
    assert all(v == 3 for v in profile_counts.values())

    for job in jobs:
        assert job.kind == "run_df_fs_sota_benchmark"
        seeds = list(job.params.get("seeds") or [])
        assert seeds == [11, 23, 37, 42, 59, 67, 73, 89, 97]


def test_validation15_reference_and_mapie_flags(tmp_path):
    jobs = build_jobs_validation15(dataset_shards=2, val14_root=Path(tmp_path))
    by_profile = {}
    for job in jobs:
        pid = _profile_from_job_id(job.job_id)
        by_profile.setdefault(pid, []).append(job)

    ref_job = by_profile["v15_ref_ipss"][0]
    assert str(ref_job.params.get("fs_method_set")) == "mnpo_v14_core_plus_ipss"
    ref_args = list(ref_job.params.get("extra_args") or [])
    assert "--regime-gating-enabled" in ref_args
    assert "--regime-gating-low-p-over-n-threshold" in ref_args
    assert ref_args[ref_args.index("--regime-gating-low-p-over-n-threshold") + 1] == "0"
    assert "--regime-gating-low-p-over-n-mode" not in ref_args

    no_regime_job = by_profile["v15_no_regime_fallback"][0]
    no_regime_args = list(no_regime_job.params.get("extra_args") or [])
    assert "--regime-gating-enabled" not in no_regime_args

    aps_args = list(by_profile["v15_mapie_aps"][0].params.get("extra_args") or [])
    assert aps_args[aps_args.index("--classifier-conformal-method") + 1] == "aps"

    raps_args = list(by_profile["v15_mapie_raps"][0].params.get("extra_args") or [])
    assert raps_args[raps_args.index("--classifier-conformal-method") + 1] == "raps"

    cross_args = list(by_profile["v15_mapie_cross"][0].params.get("extra_args") or [])
    assert cross_args[cross_args.index("--classifier-conformal-method") + 1] == "cross"


def test_validation15_bundle_sharding_keeps_profiles_together(tmp_path):
    jobs = build_jobs_validation15(dataset_shards=4, val14_root=Path(tmp_path))
    shards = _balanced_shard_assign_validation15_bundles(jobs, num_shards=4)

    assert set(shards.keys()) == {1, 2, 3, 4}
    for _, job_ids in shards.items():
        assert len(job_ids) == 9
        profiles = {_profile_from_job_id(jid) for jid in job_ids}
        assert profiles == set(VALIDATION15_PROFILE_MANIFEST.keys())
        parts = {_part_from_job_id(jid) for jid in job_ids}
        assert len(parts) == 1

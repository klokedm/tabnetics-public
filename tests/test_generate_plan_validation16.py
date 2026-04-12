from collections import Counter
from pathlib import Path

from tabnetics.validation.generate_plan import (
    VALIDATION16_DATASETS,
    VALIDATION16_PROFILE_MANIFEST,
    _balanced_shard_assign_validation16_bundles,
    build_jobs_validation16,
)


def _profile_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("/")
    assert len(parts) == 3
    return parts[1]


def _part_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("/")
    assert len(parts) == 3
    return parts[2]


def test_build_jobs_validation16_has_expected_topology(tmp_path):
    jobs = build_jobs_validation16(dataset_shards=3, val15_root=Path(tmp_path))
    assert len(jobs) == 11 * 3

    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert set(profile_counts.keys()) == set(VALIDATION16_PROFILE_MANIFEST.keys())
    assert all(v == 3 for v in profile_counts.values())

    datasets = {
        ds_id
        for job in jobs
        for ds_id in list(job.params.get("datasets") or [])
    }
    assert datasets == set(VALIDATION16_DATASETS)


def test_validation16_profile_flags_are_wired(tmp_path):
    jobs = build_jobs_validation16(dataset_shards=2, val15_root=Path(tmp_path))
    by_profile = {}
    for job in jobs:
        by_profile.setdefault(_profile_from_job_id(job.job_id), []).append(job)

    ref_job = by_profile["v16_ref"][0]
    assert str(ref_job.params.get("fs_method_set")) == "mnpo_v14_core_plus_ipss"

    clp_args = list(by_profile["v16_clp"][0].params.get("extra_args") or [])
    assert clp_args[clp_args.index("--fs-fold-preference-mode") + 1] == "logistic"

    shrink_args = list(by_profile["v16_payoff_shrink"][0].params.get("extra_args") or [])
    assert shrink_args[shrink_args.index("--fs-payoff-shrinkage-kappa") + 1] == "0.15"

    conformal_args = list(by_profile["v16_conformal_eff"][0].params.get("extra_args") or [])
    assert "--fs-use-conformal-efficiency" in conformal_args
    assert conformal_args[conformal_args.index("--fs-conformal-efficiency-method") + 1] == "aps"

    js_args = list(by_profile["v16_js_shrinkage"][0].params.get("extra_args") or [])
    assert "--fs-oracle-weight-js-shrinkage" in js_args

    meta_args = list(by_profile["v16_meta_dt"][0].params.get("extra_args") or [])
    assert meta_args[meta_args.index("--meta-learning-selector") + 1] == "decision_tree"

    multiomics_args = list(by_profile["v16_multiomics"][0].params.get("extra_args") or [])
    assert multiomics_args[multiomics_args.index("--multiomics-adapter") + 1] == "split_halves"

    full_stack_args = list(by_profile["v16_full_stack"][0].params.get("extra_args") or [])
    assert "--fs-fold-preference-mode" in full_stack_args
    assert "--fs-payoff-shrinkage-kappa" in full_stack_args
    assert "--fs-use-conformal-efficiency" in full_stack_args
    assert "--fs-oracle-weight-js-shrinkage" in full_stack_args


def test_validation16_bundle_sharding_keeps_profiles_together(tmp_path):
    jobs = build_jobs_validation16(dataset_shards=4, val15_root=Path(tmp_path))
    shards = _balanced_shard_assign_validation16_bundles(jobs, num_shards=4)

    assert set(shards.keys()) == {1, 2, 3, 4}
    for _, job_ids in shards.items():
        assert len(job_ids) == 11
        profiles = {_profile_from_job_id(jid) for jid in job_ids}
        assert profiles == set(VALIDATION16_PROFILE_MANIFEST.keys())
        parts = {_part_from_job_id(jid) for jid in job_ids}
        assert len(parts) == 1

from collections import Counter, defaultdict

import pytest

from tabnetics.validation.generate_plan import (
    BenchmarkProfile,
    VALIDATION14_PROFILE_MANIFEST,
    _assert_no_implicit_true_omissions,
    _balanced_shard_assign_validation14_activation_bundles,
    _balanced_shard_assign_validation14_bundles,
    build_jobs_validation13,
    build_jobs_validation14,
    build_jobs_validation14_activation_smoke,
)


def _profile_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("/")
    assert len(parts) == 3
    return parts[1]


def _part_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("/")
    assert len(parts) == 3
    return parts[2]


def test_build_jobs_validation14_has_expected_topology(tmp_path):
    jobs = build_jobs_validation14(dataset_shards=4, val13_root=tmp_path)
    assert len(jobs) == 20 * 4

    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert set(profile_counts.keys()) == set(VALIDATION14_PROFILE_MANIFEST.keys())
    assert all(v == 4 for v in profile_counts.values())

    for job in jobs:
        assert job.kind == "run_df_fs_sota_benchmark"
        seeds = list(job.params.get("seeds") or [])
        assert seeds == [11, 23, 37, 42, 59, 67, 73, 89, 97]


def test_build_jobs_validation14_profile_flags_are_explicit(tmp_path):
    jobs = build_jobs_validation14(dataset_shards=3, val13_root=tmp_path)
    by_profile = defaultdict(list)
    for job in jobs:
        by_profile[_profile_from_job_id(job.job_id)].append(job)

    no_bh_args = list(by_profile["v14_no_bh"][0].params.get("extra_args") or [])
    assert "--no-prefilter-bh-ttest" in no_bh_args

    no_var_args = list(by_profile["v14_no_varfloor"][0].params.get("extra_args") or [])
    assert "--no-prefilter-variance-floor" in no_var_args

    gates_only_args = list(by_profile["v14_gates_only"][0].params.get("extra_args") or [])
    assert "--no-prefilter-bh-ttest" in gates_only_args
    assert "--no-prefilter-variance-floor" in gates_only_args
    assert "--fs-copula-derandomize-runs" in gates_only_args
    assert gates_only_args[gates_only_args.index("--fs-copula-derandomize-runs") + 1] == "3"

    no_copula_args = list(by_profile["v14_no_copula5"][0].params.get("extra_args") or [])
    assert "--fs-copula-derandomize-runs" in no_copula_args
    assert no_copula_args[no_copula_args.index("--fs-copula-derandomize-runs") + 1] == "3"

    aps_args = list(by_profile["v14_mapie_aps"][0].params.get("extra_args") or [])
    assert "--classifier-conformal-method" in aps_args
    assert aps_args[aps_args.index("--classifier-conformal-method") + 1] == "aps"

    combat_args = list(by_profile["v14_batch_combat"][0].params.get("extra_args") or [])
    assert "--batch-correction" in combat_args
    assert combat_args[combat_args.index("--batch-correction") + 1] == "combat"
    assert "--batch-label-policy" in combat_args
    assert combat_args[combat_args.index("--batch-label-policy") + 1] == "kmeans2"


def test_validation14_bundle_sharding_keeps_profiles_together(tmp_path):
    jobs = build_jobs_validation14(dataset_shards=4, val13_root=tmp_path)
    shards = _balanced_shard_assign_validation14_bundles(jobs, num_shards=4)

    assert set(shards.keys()) == {1, 2, 3, 4}
    for _, job_ids in shards.items():
        assert len(job_ids) == 20
        profiles = {_profile_from_job_id(jid) for jid in job_ids}
        assert profiles == set(VALIDATION14_PROFILE_MANIFEST.keys())
        parts = {_part_from_job_id(jid) for jid in job_ids}
        assert len(parts) == 1


def test_build_jobs_validation14_activation_smoke(tmp_path):
    jobs = build_jobs_validation14_activation_smoke(dataset_shards=2, val13_root=tmp_path)
    assert len(jobs) == 20 * 2

    for job in jobs:
        assert str(job.job_id).startswith("val14_activation/")
        assert list(job.params.get("seeds") or []) == [11]
        assert int(job.params.get("task_timeout_sec") or 0) == 7200
        assert int(job.params.get("fs_method_timeout_sec") or 0) == 1800

    shards = _balanced_shard_assign_validation14_activation_bundles(jobs, num_shards=2)
    for _, job_ids in shards.items():
        profiles = {_profile_from_job_id(jid) for jid in job_ids}
        assert profiles == set(VALIDATION14_PROFILE_MANIFEST.keys())


def test_ablation_omission_detector_hard_fails():
    with pytest.raises(RuntimeError):
        _assert_no_implicit_true_omissions(
            profiles=[
                BenchmarkProfile(
                    profile_id="v14_no_bh",
                    fs_method_set="mnpo_v14_core",
                    extra_args=tuple(),
                )
            ],
            required_negations_by_profile={
                "v14_no_bh": ("--no-prefilter-bh-ttest",),
            },
            context="unit_test",
        )

    # Positive control: explicit negation passes.
    _assert_no_implicit_true_omissions(
        profiles=[
            BenchmarkProfile(
                profile_id="v14_no_bh",
                fs_method_set="mnpo_v14_core",
                extra_args=("--no-prefilter-bh-ttest",),
            )
        ],
        required_negations_by_profile={
            "v14_no_bh": ("--no-prefilter-bh-ttest",),
        },
        context="unit_test",
    )


def test_ablation_omission_detector_checks_required_value_overrides():
    with pytest.raises(RuntimeError):
        _assert_no_implicit_true_omissions(
            profiles=[
                BenchmarkProfile(
                    profile_id="v14_no_copula5",
                    fs_method_set="mnpo_v14_core",
                    extra_args=("--fs-copula-derandomize-runs", "5"),
                )
            ],
            required_value_overrides_by_profile={
                "v14_no_copula5": (("--fs-copula-derandomize-runs", "3"),),
            },
            context="unit_test_value_override",
        )

    _assert_no_implicit_true_omissions(
        profiles=[
            BenchmarkProfile(
                profile_id="v14_no_copula5",
                fs_method_set="mnpo_v14_core",
                extra_args=("--fs-copula-derandomize-runs", "3"),
            )
        ],
        required_value_overrides_by_profile={
            "v14_no_copula5": (("--fs-copula-derandomize-runs", "3"),),
        },
        context="unit_test_value_override",
    )


def test_validation13_ablation_flags_use_explicit_negations(tmp_path):
    jobs = build_jobs_validation13(dataset_shards=2, val12_root=tmp_path, val11_root=tmp_path)
    by_profile = defaultdict(list)
    for job in jobs:
        by_profile[_profile_from_job_id(job.job_id)].append(job)

    no_bh_args = list(by_profile["d_v13_no_bh"][0].params.get("extra_args") or [])
    assert "--no-prefilter-bh-ttest" in no_bh_args

    gates_only_args = list(by_profile["d_v13_gates_only"][0].params.get("extra_args") or [])
    assert "--no-prefilter-bh-ttest" in gates_only_args
    assert "--no-prefilter-variance-floor" in gates_only_args

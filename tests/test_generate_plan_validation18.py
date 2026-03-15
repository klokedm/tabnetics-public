from collections import Counter
from pathlib import Path

from tabnetics.benchmarks.runner import build_arg_parser
from tabnetics.validation.generate_plan import (
    VAL18_DIAG24,
    VAL18_FULL64,
    VAL19_ADDED_CLASSIFIERS,
    VAL19_HOST_WORKER_TARGETS,
    VAL19_HDLSS_MODERATE_CPU_NEW,
    VAL19_HDLSS_MODERATE_CPU_OLD,
    VALIDATION18_ANCHORS_PROFILE_MANIFEST,
    VALIDATION18_CLASSIFIERS_PROFILE_MANIFEST,
    VALIDATION18_MNPO_PROFILE_MANIFEST,
    VALIDATION18_STAGE_PROFILE_MANIFEST,
    VALIDATION19_CLASSIFIERS_PROFILE_MANIFEST,
    _balanced_shard_assign_validation19_classifiers_bundles,
    _validation19_recommended_host_assignment,
    build_jobs_validation18_anchors,
    _balanced_shard_assign_validation18_classifiers_bundles,
    build_jobs_validation18_classifiers,
    build_jobs_validation19_classifiers,
    build_jobs_validation18_mnpo,
    build_jobs_validation18_stage,
)


def _profile_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("/")
    assert len(parts) == 3
    return parts[1]


def _extra_args_by_flag(job) -> dict[str, str]:
    extra_args = list(job.params.get("extra_args") or [])
    out: dict[str, str] = {}
    i = 0
    while i < len(extra_args):
        item = str(extra_args[i])
        if not item.startswith("--"):
            i += 1
            continue
        if i + 1 < len(extra_args) and not str(extra_args[i + 1]).startswith("--"):
            out[item] = str(extra_args[i + 1])
            i += 2
            continue
        out[item] = ""
        i += 1
    return out


def _candidate_list(job) -> list[str]:
    extra_args = list(job.params.get("extra_args") or [])
    idx = len(extra_args) - 1 - extra_args[::-1].index("--model-candidates")
    end = len(extra_args) - 1 - extra_args[::-1].index("--model-cv-runtime-max-candidates")
    idx += 1
    return [str(x) for x in extra_args[idx:end]]


def test_build_jobs_validation18_mnpo_matches_val18_plan_profiles(tmp_path):
    jobs = build_jobs_validation18_mnpo(dataset_shards=2, val17_root=Path(tmp_path))
    assert len(jobs) == len(VALIDATION18_MNPO_PROFILE_MANIFEST) * 2

    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert set(profile_counts.keys()) == set(VALIDATION18_MNPO_PROFILE_MANIFEST.keys())
    assert all(v == 2 for v in profile_counts.values())

    by_profile = {profile: [j for j in jobs if _profile_from_job_id(j.job_id) == profile] for profile in profile_counts}
    n30_args = _extra_args_by_flag(by_profile["N30_core_banzhaf_perf_only"][0])
    assert "--disable-fs-stability-oracle" in n30_args
    assert "--disable-fs-complexity-oracle" in n30_args
    assert "--disable-fs-robust-oracle" in n30_args
    assert "--enable-diversity-oracle" not in n30_args

    n31_args = _extra_args_by_flag(by_profile["N31_core_banzhaf_perf_complexity"][0])
    assert "--disable-fs-stability-oracle" in n31_args
    assert "--disable-fs-complexity-oracle" not in n31_args
    assert "--disable-fs-robust-oracle" in n31_args

    n40_args = _extra_args_by_flag(by_profile["N40_core_banzhaf_5x5"][0])
    assert n40_args["--fs-inner-cv-splits"] == "5"
    assert n40_args["--fs-inner-cv-repeats"] == "5"

    n06_args = _extra_args_by_flag(by_profile["N06_core_banzhaf_no_clp"][0])
    assert n06_args["--fs-fold-preference-mode"] == "vote"

    n07_args = _extra_args_by_flag(by_profile["N07_core_banzhaf_no_payoff"][0])
    assert n07_args["--fs-payoff-shrinkage-kappa"] == "0.0"

    n08_args = _extra_args_by_flag(by_profile["N08_core_banzhaf_no_clp_payoff"][0])
    assert n08_args["--fs-fold-preference-mode"] == "vote"
    assert n08_args["--fs-payoff-shrinkage-kappa"] == "0.0"

    n09_args = _extra_args_by_flag(by_profile["N09_core_banzhaf_no_js"][0])
    assert "--fs-oracle-weight-js-shrinkage" not in n09_args

    n10_args = _extra_args_by_flag(by_profile["N10_core_banzhaf_no_conformal_eff"][0])
    assert "--fs-use-conformal-efficiency" not in n10_args
    assert "--fs-conformal-efficiency-method" not in n10_args


def test_build_jobs_validation18_stage_matches_live_val18_plan(tmp_path):
    jobs = build_jobs_validation18_stage(dataset_shards=2, val17_root=Path(tmp_path))
    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert set(profile_counts.keys()) == set(VALIDATION18_STAGE_PROFILE_MANIFEST.keys())
    assert "P17_multiomics_mb_plsda" not in profile_counts
    assert "P18_multiomics_mint" not in profile_counts
    assert "S01_diablo_blocks" not in profile_counts
    assert profile_counts["S02_batch_combat_seq"] == 2
    assert profile_counts["S03_cpss_irp_bounded"] == 2


def test_build_jobs_validation18_classifiers_adds_tabpfn_full64_lane(tmp_path):
    jobs = build_jobs_validation18_classifiers(dataset_shards=2, val17_root=Path(tmp_path))
    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert set(profile_counts.keys()) == set(VALIDATION18_CLASSIFIERS_PROFILE_MANIFEST.keys())
    assert profile_counts["C10_only_tabpfn_full64"] == 2
    assert profile_counts["C11_pool_legacy_plus_tabpfn_full64"] == 2
    assert profile_counts["C12_pool_mnpo_plus_tabpfn_full64"] == 2

    c10_jobs = [j for j in jobs if _profile_from_job_id(j.job_id) == "C10_only_tabpfn_full64"]
    assert all(str(j.params.get("execution_lane")) == "tabpfn" for j in c10_jobs)
    assert all(str(j.params.get("dataset_panel")) == "full64" for j in c10_jobs)
    assert all(list(j.params.get("preferred_hosts") or []) == ["arch-ml"] for j in c10_jobs)
    full64_seen = {ds for job in c10_jobs for ds in list(job.params.get("datasets") or [])}
    assert full64_seen == set(VAL18_FULL64)

    c01_jobs = [j for j in jobs if _profile_from_job_id(j.job_id) == "C01_pool_legacy_core"]
    assert all(str(j.params.get("execution_lane")) == "cpu" for j in c01_jobs)
    assert all(str(j.params.get("dataset_panel")) == "diag24" for j in c01_jobs)
    assert all(
        list(j.params.get("preferred_hosts") or [])
        == ["host0.example.com", "host0.example.com", "host0.example.com", "host0.example.com"]
        for j in c01_jobs
    )
    diag_seen = {ds for job in c01_jobs for ds in list(job.params.get("datasets") or [])}
    assert diag_seen == set(VAL18_DIAG24)


def test_build_jobs_validation19_classifiers_matches_plan_profiles(tmp_path):
    jobs = build_jobs_validation19_classifiers(dataset_shards=2, val17_root=Path(tmp_path))
    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert set(profile_counts.keys()) == set(VALIDATION19_CLASSIFIERS_PROFILE_MANIFEST.keys())
    assert profile_counts["C_ONLY_cpda"] == 2
    assert profile_counts["V19_C01_old_regime_legacy_full64"] == 2
    assert profile_counts["V19_C04_new_regime_mnpo_full64"] == 2
    assert profile_counts["V19_C05_old_regime_mnpo_val18compat_full64"] == 2
    assert profile_counts["V19_C06_new_regime_mnpo_val18compat_full64"] == 2

    cpda_job = next(j for j in jobs if _profile_from_job_id(j.job_id) == "C_ONLY_cpda")
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--datasets",
            "colon_alon",
            "--fs-method-set",
            "mnpo_v14_core_plus_ipss",
            "--output-dir",
            str(tmp_path / "v19_cpda_parse"),
            *list(cpda_job.params.get("extra_args") or []),
        ]
    )
    assert "cpda" in list(getattr(args, "model_candidates", []) or [])


def test_validation19_old_and_new_pool_snapshots_are_frozen_in_profile_args(tmp_path):
    jobs = build_jobs_validation19_classifiers(dataset_shards=2, val17_root=Path(tmp_path))
    old_job = next(j for j in jobs if _profile_from_job_id(j.job_id) == "V19_C01_old_regime_legacy_full64")
    new_job = next(j for j in jobs if _profile_from_job_id(j.job_id) == "V19_C02_new_regime_legacy_full64")

    old_candidates = _candidate_list(old_job)
    new_candidates = _candidate_list(new_job)
    assert tuple(old_candidates) == VAL19_HDLSS_MODERATE_CPU_OLD
    assert tuple(new_candidates) == VAL19_HDLSS_MODERATE_CPU_NEW
    assert "cpda" not in old_candidates
    assert "cpda" in new_candidates
    assert "tabpfn" not in old_candidates
    assert "tabpfn" not in new_candidates
    assert all(c in new_candidates for c in VAL19_ADDED_CLASSIFIERS)
    assert str(old_job.params.get("dataset_panel")) == "full64"
    assert str(new_job.params.get("pool_snapshot_id")) == "new"


def test_validation19_oracle_compat_profiles_parse_cleanly(tmp_path):
    jobs = build_jobs_validation19_classifiers(dataset_shards=2, val17_root=Path(tmp_path))
    parser = build_arg_parser()

    compat_job = next(
        j for j in jobs if _profile_from_job_id(j.job_id) == "V19_C05_old_regime_mnpo_val18compat_full64"
    )
    compat_args = list(compat_job.params.get("extra_args") or [])
    parsed = parser.parse_args(
        [
            "--datasets",
            "colon_alon",
            "--fs-method-set",
            "mnpo_v14_core_plus_ipss",
            "--output-dir",
            str(tmp_path / "v19_oracle_compat_parse"),
            *compat_args,
        ]
    )
    assert str(getattr(parsed, "classifier_oracle_behavior_profile", "")) == "val18_compat"

    current_job = next(j for j in jobs if _profile_from_job_id(j.job_id) == "V19_C03_old_regime_mnpo_full64")
    current_args = _extra_args_by_flag(current_job)
    assert current_args["--classifier-oracle-behavior-profile"] == "current"


def test_validation19_classifier_sharding_keeps_jobs_complete(tmp_path):
    jobs = build_jobs_validation19_classifiers(dataset_shards=2, val17_root=Path(tmp_path))
    shards = _balanced_shard_assign_validation19_classifiers_bundles(jobs, num_shards=4)
    flattened = [jid for shard in shards.values() for jid in shard]
    assert sorted(flattened) == sorted(j.job_id for j in jobs)


def test_validation19_host_assignment_uses_conservative_cpu_only_worker_targets(tmp_path):
    jobs = build_jobs_validation19_classifiers(dataset_shards=2, val17_root=Path(tmp_path))
    shards = _balanced_shard_assign_validation19_classifiers_bundles(jobs, num_shards=4)
    jobs_by_id = {j.job_id: j for j in jobs}

    assignment = _validation19_recommended_host_assignment(shards, jobs_by_id)
    host_summary = dict(assignment.get("host_summary") or {})

    assert "arch-ml" not in host_summary
    assert set(host_summary.keys()) == set(VAL19_HOST_WORKER_TARGETS.keys())
    for host, expected in VAL19_HOST_WORKER_TARGETS.items():
        assert dict(host_summary[host]["worker_target"]) == dict(expected)
        assert int(host_summary[host]["worker_target"]["max_workers_per_pod"]) <= 4


def test_validation18_c05_conformal_off_args_parse_cleanly(tmp_path):
    jobs = build_jobs_validation18_classifiers(dataset_shards=2, val17_root=Path(tmp_path))
    c05_job = next(j for j in jobs if _profile_from_job_id(j.job_id) == "C05_conformal_off")
    extra_args = list(c05_job.params.get("extra_args") or [])

    assert "--enable-classifier-conformal" not in extra_args
    assert "--classifier-conformal-alpha" not in extra_args
    assert "--classifier-conformal-calibration-fraction" not in extra_args
    assert "--classifier-conformal-min-calibration" not in extra_args
    assert "--classifier-conformal-method" not in extra_args

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--datasets",
            "colon_alon",
            "--fs-method-set",
            "mnpo_v14_core_plus_ipss",
            "--output-dir",
            str(tmp_path / "c05_parse"),
            *extra_args,
        ]
    )
    assert bool(getattr(args, "enable_classifier_conformal", False)) is False


def test_validation18_sglnn_profiles_parse_cleanly(tmp_path):
    jobs = build_jobs_validation18_classifiers(dataset_shards=2, val17_root=Path(tmp_path))
    parser = build_arg_parser()

    for profile_id in ("C_ONLY_sglnn", "S08_sglnn"):
        job = next(j for j in jobs if _profile_from_job_id(j.job_id) == profile_id)
        extra_args = list(job.params.get("extra_args") or [])
        args = parser.parse_args(
            [
                "--datasets",
                "colon_alon",
                "--fs-method-set",
                "mnpo_v14_core_plus_ipss",
                "--output-dir",
                str(tmp_path / profile_id),
                *extra_args,
            ]
        )
        assert "sglnn" in list(getattr(args, "model_candidates", []) or [])

    include_args = parser.parse_args(
        [
            "--datasets",
            "colon_alon",
            "--fs-method-set",
            "mnpo_v14_core_plus_ipss",
            "--output-dir",
            str(tmp_path / "sglnn_include"),
            "--include-sglnn-model",
        ]
    )
    assert bool(getattr(include_args, "include_sglnn_model", False)) is True


def test_validation18_anchor_and_stage_profiles_parse_cleanly(tmp_path):
    parser = build_arg_parser()

    anchor_jobs = build_jobs_validation18_anchors(dataset_shards=2, val17_root=Path(tmp_path))
    anchor_counts = Counter(_profile_from_job_id(j.job_id) for j in anchor_jobs)
    assert set(anchor_counts.keys()) == set(VALIDATION18_ANCHORS_PROFILE_MANIFEST.keys())
    a08_job = next(j for j in anchor_jobs if _profile_from_job_id(j.job_id) == "A08_ref_no_regime_gating")
    a08_args = list(a08_job.params.get("extra_args") or [])
    assert not any(str(arg).startswith("--regime-gating") for arg in a08_args)
    parser.parse_args(
        [
            "--datasets",
            "colon_alon",
            "--fs-method-set",
            "mnpo_v14_core_plus_ipss",
            "--output-dir",
            str(tmp_path / "a08_parse"),
            *a08_args,
        ]
    )

    stage_jobs = build_jobs_validation18_stage(dataset_shards=2, val17_root=Path(tmp_path))
    expected_stage_profiles = {
        "D14_batch_combat_kmeans2": ("--batch-label-policy", "kmeans2"),
        "D17_multimodal_gmm": ("--df-multimodal-fallback", "gmm"),
        "D18_multimodal_rank": ("--df-multimodal-fallback", "rank_transform"),
        "P05_rank_prefilter_off": ("--disable-rank-prefilter", ""),
        "P13_regime_off": (None, None),
        "P14_lowpn_fast_filter": ("--regime-gating-low-p-over-n-mode", "fast_univariate_filter"),
        "P15_lowpn_all_features": ("--regime-gating-low-p-over-n-mode", "all_features"),
    }
    for profile_id, expected in expected_stage_profiles.items():
        job = next(j for j in stage_jobs if _profile_from_job_id(j.job_id) == profile_id)
        extra_args = list(job.params.get("extra_args") or [])
        if expected[0] is not None:
            assert expected[0] in extra_args
            if expected[1]:
                idx = extra_args.index(expected[0]) + 1
                assert extra_args[idx] == expected[1]
        if profile_id == "P13_regime_off":
            assert not any(str(arg).startswith("--regime-gating") for arg in extra_args)
        parser.parse_args(
            [
                "--datasets",
                "colon_alon",
                "--fs-method-set",
                "mnpo_v14_core_plus_ipss",
                "--output-dir",
                str(tmp_path / f"{profile_id}_parse"),
                *extra_args,
            ]
        )


def test_build_jobs_validation18_stage_default_uses_seven_dataset_shards(tmp_path):
    jobs = build_jobs_validation18_stage(val17_root=Path(tmp_path))
    assert len(jobs) == len(VALIDATION18_STAGE_PROFILE_MANIFEST) * 7


def test_validation18_classifier_sharding_keeps_cpu_and_tabpfn_separate(tmp_path):
    jobs = build_jobs_validation18_classifiers(dataset_shards=2, val17_root=Path(tmp_path))
    shards = _balanced_shard_assign_validation18_classifiers_bundles(jobs, num_shards=4)
    jobs_by_id = {j.job_id: j for j in jobs}

    shard_lanes = []
    for job_ids in shards.values():
        lanes = {str(jobs_by_id[jid].params.get("execution_lane")) for jid in job_ids}
        lanes.discard("")
        if lanes:
            shard_lanes.append(lanes)
            assert len(lanes) == 1
    assert {"cpu"} in shard_lanes
    assert {"tabpfn"} in shard_lanes


def test_runner_parser_accepts_val18_oracle_pruning_flags(tmp_path):
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--datasets",
            "colon_alon",
            "--fs-method-set",
            "mnpo_v14_core_plus_ipss",
            "--output-dir",
            str(tmp_path),
            "--disable-fs-stability-oracle",
            "--disable-fs-complexity-oracle",
            "--disable-fs-robust-oracle",
            "--fs-inner-cv-splits",
            "5",
            "--fs-inner-cv-repeats",
            "5",
        ]
    )
    assert args.disable_fs_stability_oracle is True
    assert args.disable_fs_complexity_oracle is True
    assert args.disable_fs_robust_oracle is True
    assert int(args.fs_inner_cv_splits) == 5
    assert int(args.fs_inner_cv_repeats) == 5

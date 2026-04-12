import json
from collections import Counter
from pathlib import Path

import pytest

from tabnetics.benchmarks.runner import build_arg_parser
from tabnetics.validation.generate_plan import (
    VAL18_DIAG24,
    VAL18_FULL64,
    VAL19_ADDED_CLASSIFIERS,
    VAL19_HOST_WORKER_TARGETS,
    VAL19_HDLSS_MODERATE_CPU_NEW,
    VAL19_HDLSS_MODERATE_CPU_OLD,
    VAL20_CORE_CLASSIFIER_POOL,
    VAL20_EXPANDED_CLASSIFIER_POOL,
    VAL20_EXPANDED_CLASSIFIER_POOL_NO_TABPFN,
    VAL20_HOST_WORKER_TARGETS,
    VAL20_TABARENA_HOSTS,
    VAL21_RV_HOLDOUT,
    VALIDATION21_PHASE1_PROFILE_MANIFEST,
    VALIDATION18_ANCHORS_PROFILE_MANIFEST,
    VALIDATION18_CLASSIFIERS_PROFILE_MANIFEST,
    VALIDATION18_MNPO_PROFILE_MANIFEST,
    VALIDATION18_STAGE_PROFILE_MANIFEST,
    VALIDATION19_CLASSIFIERS_PROFILE_MANIFEST,
    VALIDATION20_EXTENSION_PROFILE_MANIFEST,
    VALIDATION20_TABARENA_W1_PROFILE_MANIFEST,
    VALIDATION20_TABARENA_W2_PROFILE_MANIFEST,
    VALIDATION20_TABARENA_W3_PROFILE_MANIFEST,
    VALIDATION20_WAVE2_RESERVED_PROFILE_MANIFEST,
    VALIDATION20_WAVE3_RESERVED_PROFILE_MANIFEST,
    VALIDATION20_WAVE1_PROFILE_MANIFEST,
    _balanced_shard_assign_validation20_tabarena_bundles,
    _balanced_shard_assign_validation20_ensemble_bundles,
    _balanced_shard_assign_validation20_wave1_bundles,
    _balanced_shard_assign_validation21_phase1_bundles,
    _balanced_shard_assign_validation21_phase2_bundles,
    _validation20_tabarena_recommended_host_assignment,
    _validation20_recommended_host_assignment,
    _validation21_recommended_host_assignment,
    _balanced_shard_assign_validation19_classifiers_bundles,
    _validation19_recommended_host_assignment,
    build_jobs_validation18_anchors,
    _balanced_shard_assign_validation18_classifiers_bundles,
    build_jobs_validation18_classifiers,
    build_jobs_validation19_classifiers,
    build_jobs_validation20_ensemble,
    build_jobs_validation20_tabarena_w1,
    build_jobs_validation20_tabarena_w2,
    build_jobs_validation20_tabarena_w3,
    build_jobs_validation20_wave1,
    build_jobs_validation21_phase1,
    build_jobs_validation21_phase2,
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


def _write_val21_spec(tmp_path: Path, *, phase2_profile_ids: list[str] | None = None) -> Path:
    spec = {
        "schema_version": 1,
        "winner_profile_id": "V20_C02_candidate_b_full64",
        "winner_fs_method_set": "strict_plus_mrmr",
        "winner_execution_lane": "cpu",
        "winner_extra_args": [
            "--classifier-selection-mode",
            "legacy",
            "--classification-backend",
            "sklearn",
            "--model-candidates",
            "lr",
            "--model-cv-runtime-max-candidates",
            "1",
        ],
        "current_default_profile_id": "V20_C04_current_default_full64",
        "current_default_fs_method_set": "strict_plus_mrmr",
        "current_default_execution_lane": "cpu",
        "current_default_extra_args": [
            "--classifier-selection-mode",
            "legacy",
            "--classification-backend",
            "sklearn",
            "--model-candidates",
            "lr",
            "--model-cv-runtime-max-candidates",
            "1",
        ],
        "lockout_fallback_fs_method_set": "strict_plus_mrmr",
        "meta_learning_records_path": "/tmp/val21_meta_learning_records.json",
        "tier_classifier_model_path": "/tmp/tier_classifier_model.json",
        "phase2_profile_ids": phase2_profile_ids or ["V21_WINNER", "V21_INTEGRATED", "V21_CURRENT_DEFAULT"],
    }
    path = tmp_path / "val21_spec.json"
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return path


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


def test_build_jobs_validation20_wave1_matches_plan_profiles(tmp_path):
    jobs = build_jobs_validation20_wave1(dataset_shards=2, val17_root=Path(tmp_path))
    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert set(profile_counts.keys()) == set(VALIDATION20_WAVE1_PROFILE_MANIFEST.keys())
    assert not (set(profile_counts.keys()) & set(VALIDATION20_WAVE2_RESERVED_PROFILE_MANIFEST.keys()))
    assert not (set(profile_counts.keys()) & set(VALIDATION20_WAVE3_RESERVED_PROFILE_MANIFEST.keys()))
    assert profile_counts["V20_B03_mnpo_ref_anchor"] == 2
    assert profile_counts["V20_F04_sklearn_mnpo_diag24"] == 2
    assert profile_counts["V20_T03_tabpfn_extreme_only_mnpo_full64"] == 2
    assert profile_counts["V20_E07_all_enhancements_diag24"] == 2

    b03_job = next(j for j in jobs if _profile_from_job_id(j.job_id) == "V20_B03_mnpo_ref_anchor")
    b03_candidates = _candidate_list(b03_job)
    assert tuple(b03_candidates) == VAL20_CORE_CLASSIFIER_POOL
    assert str(b03_job.params.get("execution_lane")) == "cpu"

    b05_job = next(j for j in jobs if _profile_from_job_id(j.job_id) == "V20_B05_current_pool_mnpo")
    b05_candidates = _candidate_list(b05_job)
    assert tuple(b05_candidates) == VAL20_EXPANDED_CLASSIFIER_POOL
    assert str(b05_job.params.get("execution_lane")) == "tabpfn"
    assert list(b05_job.params.get("preferred_hosts") or []) == ["arch-ml"]

    t03_args = _extra_args_by_flag(
        next(j for j in jobs if _profile_from_job_id(j.job_id) == "V20_T03_tabpfn_extreme_only_mnpo_full64")
    )
    assert t03_args["--classifier-regime-candidate-exclusions"] == "hdlss_moderate:tabpfn"


def test_validation20_reserved_follow_on_profile_manifests_are_explicit_and_separate() -> None:
    assert set(VALIDATION20_WAVE2_RESERVED_PROFILE_MANIFEST.keys()) == {
        "V20_F05_flaml_best_mnpo_full64",
        "V20_F06_flaml_best_legacy_full64",
    }
    assert set(VALIDATION20_WAVE3_RESERVED_PROFILE_MANIFEST.keys()) == {
        "V20_C01_candidate_a_full64",
        "V20_C02_candidate_b_full64",
        "V20_C03_candidate_c_full64",
        "V20_C04_current_default_full64",
    }
    assert not (set(VALIDATION20_WAVE1_PROFILE_MANIFEST.keys()) & set(VALIDATION20_WAVE2_RESERVED_PROFILE_MANIFEST.keys()))
    assert not (set(VALIDATION20_WAVE1_PROFILE_MANIFEST.keys()) & set(VALIDATION20_WAVE3_RESERVED_PROFILE_MANIFEST.keys()))


def test_validation20_wave1_host_assignment_uses_three_host_policy(tmp_path):
    jobs = build_jobs_validation20_wave1(dataset_shards=2, val17_root=Path(tmp_path))
    shards = _balanced_shard_assign_validation20_wave1_bundles(jobs, num_shards=8)
    flattened = [jid for shard in shards.values() for jid in shard]
    assert sorted(flattened) == sorted(j.job_id for j in jobs)

    jobs_by_id = {j.job_id: j for j in jobs}
    assignment = _validation20_recommended_host_assignment(shards, jobs_by_id)
    host_summary = dict(assignment.get("host_summary") or {})

    assert set(host_summary.keys()) == set(VAL20_HOST_WORKER_TARGETS.keys())
    assert "arch-ml" in host_summary
    for host, expected in VAL20_HOST_WORKER_TARGETS.items():
        assert dict(host_summary[host]["worker_target"]) == dict(expected)


def test_build_jobs_validation20_ensemble_includes_classifier_mnpo_diagnostics(tmp_path):
    jobs = build_jobs_validation20_ensemble(dataset_shards=2, val17_root=Path(tmp_path))
    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert set(profile_counts.keys()) == set(VALIDATION20_EXTENSION_PROFILE_MANIFEST.keys())
    assert profile_counts["V20_O01_current_tritrust_diag24"] == 2
    assert profile_counts["V20_O06_top3_no_diversity_diag24"] == 2

    shards = _balanced_shard_assign_validation20_ensemble_bundles(jobs, num_shards=4)
    flattened = [jid for shard in shards.values() for jid in shard]
    assert sorted(flattened) == sorted(j.job_id for j in jobs)

    o01_job = next(j for j in jobs if _profile_from_job_id(j.job_id) == "V20_O01_current_tritrust_diag24")
    assert tuple(_candidate_list(o01_job)) == VAL20_EXPANDED_CLASSIFIER_POOL_NO_TABPFN
    assert str(o01_job.params.get("execution_lane")) == "cpu"
    assert str(o01_job.params.get("diagnostic_focus")) == "classifier_mnpo"

    o05_args = _extra_args_by_flag(
        next(j for j in jobs if _profile_from_job_id(j.job_id) == "V20_O05_val18compat_tritrust_diag24")
    )
    assert o05_args["--classifier-oracle-behavior-profile"] == "val18_compat"

    o06_args = _extra_args_by_flag(
        next(j for j in jobs if _profile_from_job_id(j.job_id) == "V20_O06_top3_no_diversity_diag24")
    )
    assert o06_args["--classifier-oracle-k"] == "3"
    assert "--enable-classifier-oracle-portfolio-diversity" not in o06_args


def test_validation21_phase1_requires_resolved_spec(tmp_path):
    with pytest.raises(RuntimeError, match="unresolved placeholders"):
        build_jobs_validation21_phase1(
            dataset_shards=2,
            val17_root=Path(tmp_path),
        )


def test_build_jobs_validation21_phase1_matches_manifest(tmp_path):
    spec_path = _write_val21_spec(Path(tmp_path))
    jobs = build_jobs_validation21_phase1(
        dataset_shards=2,
        val17_root=Path(tmp_path),
        spec_path=spec_path,
    )
    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert set(profile_counts.keys()) == set(VALIDATION21_PHASE1_PROFILE_MANIFEST.keys())
    assert all(v == 2 for v in profile_counts.values())

    lockout_job = next(j for j in jobs if _profile_from_job_id(j.job_id) == "V21_WINNER_LOCKOUT")
    lockout_args = _extra_args_by_flag(lockout_job)
    assert lockout_args["--tier-lockout-difficulty-source"] == "meta_features"
    assert lockout_args["--tier-classifier-mode"] == "learned"
    assert lockout_args["--tier-classifier-model-path"] == "/tmp/tier_classifier_model.json"
    assert list(lockout_job.params.get("preferred_hosts") or []) == ["host0.example.com", "host0.example.com"]

    metasel_job = next(j for j in jobs if _profile_from_job_id(j.job_id) == "V21_WINNER_METASEL")
    metasel_args = _extra_args_by_flag(metasel_job)
    assert metasel_args["--meta-learning-selector"] == "decision_tree"
    assert metasel_args["--meta-learning-records-path"] == "/tmp/val21_meta_learning_records.json"

    phase1_shards = _balanced_shard_assign_validation21_phase1_bundles(jobs, num_shards=4)
    flattened = [jid for shard in phase1_shards.values() for jid in shard]
    assert sorted(flattened) == sorted(j.job_id for j in jobs)

    assignment = _validation21_recommended_host_assignment(phase1_shards, {j.job_id: j for j in jobs})
    assert set((assignment.get("host_summary") or {}).keys()) == {"host0.example.com", "host0.example.com", "arch-ml"}


def test_build_jobs_validation21_phase2_uses_selected_profiles_and_rv_holdout(tmp_path):
    spec_path = _write_val21_spec(
        Path(tmp_path),
        phase2_profile_ids=["V21_WINNER", "V21_INTEGRATED", "V21_CURRENT_DEFAULT"],
    )
    jobs = build_jobs_validation21_phase2(
        dataset_shards=2,
        val17_root=Path(tmp_path),
        spec_path=spec_path,
    )
    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)
    assert set(profile_counts.keys()) == {"V21_WINNER", "V21_INTEGRATED", "V21_CURRENT_DEFAULT"}
    assert all(v == 2 for v in profile_counts.values())
    for job in jobs:
        assert list(job.params.get("datasets") or [])
        assert set(job.params.get("datasets") or []).issubset(set(VAL21_RV_HOLDOUT))

    phase2_shards = _balanced_shard_assign_validation21_phase2_bundles(jobs, num_shards=2)
    flattened = [jid for shard in phase2_shards.values() for jid in shard]
    assert sorted(flattened) == sorted(j.job_id for j in jobs)


def test_build_jobs_validation20_tabarena_w1_matches_plan_profiles() -> None:
    jobs = build_jobs_validation20_tabarena_w1(dataset_shards=2)
    profile_counts = Counter(_profile_from_job_id(j.job_id) for j in jobs)

    assert set(profile_counts.keys()) == set(VALIDATION20_TABARENA_W1_PROFILE_MANIFEST.keys())
    assert profile_counts["TA_W1_A_general_tabular_probe_refresh"] == 2
    assert profile_counts["TA_W1_B_general_tabular_competitive_probe"] == 2
    assert all(j.kind == "tabarena_benchmark" for j in jobs)

    baseline_job = next(j for j in jobs if _profile_from_job_id(j.job_id) == "TA_W1_A_general_tabular_probe_refresh")
    challenger_job = next(
        j for j in jobs if _profile_from_job_id(j.job_id) == "TA_W1_B_general_tabular_competitive_probe"
    )

    assert list(baseline_job.params.get("dataset_sets") or []) == ["general_tabular_probe"]
    assert list(baseline_job.params.get("seeds") or []) == [42, 52, 62]
    assert str(baseline_job.params.get("profile")) == "general_tabular"
    assert int(baseline_job.params.get("official_fold_limit") or -1) == 2
    assert bool(baseline_job.params.get("skip_official_leaderboard", False)) is True
    assert list(baseline_job.params.get("preferred_hosts") or []) == list(VAL20_TABARENA_HOSTS)

    baseline_args = _extra_args_by_flag(baseline_job)
    challenger_args = _extra_args_by_flag(challenger_job)
    assert baseline_args["--flaml-time-budget"] == "75"
    assert challenger_args["--flaml-time-budget"] == "75"
    assert str(challenger_job.params.get("profile")) == "general_tabular_competitive"


def test_build_jobs_validation20_tabarena_w2_and_w3_match_plan_profiles() -> None:
    w2_jobs = build_jobs_validation20_tabarena_w2(dataset_shards=2)
    w2_counts = Counter(_profile_from_job_id(j.job_id) for j in w2_jobs)
    assert set(w2_counts.keys()) == set(VALIDATION20_TABARENA_W2_PROFILE_MANIFEST.keys())
    assert all(v == 2 for v in w2_counts.values())

    w2_baseline = next(j for j in w2_jobs if _profile_from_job_id(j.job_id) == "TA_W2_A_general_tabular_full_refresh")
    w2_promoted = next(j for j in w2_jobs if _profile_from_job_id(j.job_id) == "TA_W2_C_val20_promoted_challenger")
    assert list(w2_baseline.params.get("dataset_sets") or []) == ["all"]
    assert list(w2_baseline.params.get("seeds") or []) == [42]
    assert int(w2_baseline.params.get("official_fold_limit", -1)) == 0
    assert bool(w2_baseline.params.get("skip_official_leaderboard", True)) is False
    assert str(w2_baseline.params.get("leaderboard_method_name")) == "tabnetics_general_tabular_refresh"
    assert "--enable-classifier-oracle-cvar" in list(w2_promoted.params.get("extra_args") or [])

    w3_jobs = build_jobs_validation20_tabarena_w3(dataset_shards=2)
    w3_counts = Counter(_profile_from_job_id(j.job_id) for j in w3_jobs)
    assert set(w3_counts.keys()) == set(VALIDATION20_TABARENA_W3_PROFILE_MANIFEST.keys())
    assert all(v == 2 for v in w3_counts.values())
    assert all(bool(j.params.get("skip_official_leaderboard", True)) is False for j in w3_jobs)


def test_validation20_tabarena_host_assignment_uses_three_host_cpu_policy() -> None:
    jobs = build_jobs_validation20_tabarena_w1(dataset_shards=2)
    shards = _balanced_shard_assign_validation20_tabarena_bundles(jobs, num_shards=4)
    flattened = [jid for shard in shards.values() for jid in shard]
    assert sorted(flattened) == sorted(j.job_id for j in jobs)

    jobs_by_id = {j.job_id: j for j in jobs}
    assignment = _validation20_tabarena_recommended_host_assignment(shards, jobs_by_id)
    host_summary = dict(assignment.get("host_summary") or {})

    assert set(host_summary.keys()) == set(VAL20_HOST_WORKER_TARGETS.keys())
    assert "arch-ml" in host_summary
    for host, expected in VAL20_HOST_WORKER_TARGETS.items():
        assert dict(host_summary[host]["worker_target"]) == dict(expected)


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

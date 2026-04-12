from pathlib import Path

from tabnetics.validation.core.shard_runner import _build_tabarena_benchmark_cmd


def test_build_tabarena_benchmark_cmd_wires_profile_shards_and_leaderboard_flags(tmp_path: Path) -> None:
    job = {
        "job_id": "val20_tabarena_w2/TA_W2_B_competitive_full64/ts03",
        "kind": "tabarena_benchmark",
        "params": {
            "dataset_sets": ["all"],
            "datasets": [],
            "exclude_datasets": [],
            "seeds": [42],
            "profile": "general_tabular_competitive",
            "protocol": "openml_task",
            "official_fold_limit": 0,
            "task_shard_count": 8,
            "task_shard_index": 2,
            "task_timeout_sec": 14400.0,
            "progress_heartbeat_sec": 30.0,
            "progress_watchdog_sec": 0.0,
            "progress_stall_watchdog_sec": 1800.0,
            "leaderboard_method_name": "tabnetics_general_tabular_competitive",
            "skip_official_leaderboard": False,
            "extra_args": ["--enable-classifier-oracle-cvar"],
        },
    }

    cmd = _build_tabarena_benchmark_cmd(job, out_dir=tmp_path / "out", max_workers=5)

    assert cmd[1:3] == ["-m", "experiments.benchmarking.tabarena_benchmark"]
    assert "--dataset-sets" in cmd
    assert "--profile" in cmd and cmd[cmd.index("--profile") + 1] == "general_tabular_competitive"
    assert "--task-shard-count" in cmd and cmd[cmd.index("--task-shard-count") + 1] == "8"
    assert "--task-shard-index" in cmd and cmd[cmd.index("--task-shard-index") + 1] == "2"
    assert "--leaderboard-method-name" in cmd
    assert "tabnetics_general_tabular_competitive" in cmd
    assert "--enable-classifier-oracle-cvar" in cmd
    assert "--skip-official-leaderboard" not in cmd


def test_build_tabarena_benchmark_cmd_can_disable_leaderboard(tmp_path: Path) -> None:
    job = {
        "job_id": "val20_tabarena_w1/TA_W1_A_general_tabular_probe_refresh/ts01",
        "kind": "tabarena_benchmark",
        "params": {
            "dataset_sets": ["general_tabular_probe"],
            "seeds": [42, 52, 62],
            "profile": "general_tabular",
            "protocol": "openml_task",
            "official_fold_limit": 2,
            "task_shard_count": 4,
            "task_shard_index": 0,
            "skip_official_leaderboard": True,
            "quiet": True,
            "extra_args": ["--flaml-time-budget", "75"],
        },
    }

    cmd = _build_tabarena_benchmark_cmd(job, out_dir=tmp_path / "out", max_workers=4)

    assert "--skip-official-leaderboard" in cmd
    assert "--quiet" in cmd
    assert "--flaml-time-budget" in cmd and cmd[cmd.index("--flaml-time-budget") + 1] == "75"

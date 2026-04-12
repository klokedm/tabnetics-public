from tabnetics.benchmarks import runner as benchmark
from tabnetics.datasets import benchmark_catalog as dsmod
from tabnetics.benchmarks.config import build_base_config


def test_dataset_module_exports_match_runner_bindings():
    assert set(benchmark.BENCHMARK_DATASETS.keys()) == set(dsmod.BENCHMARK_DATASETS.keys())
    assert set(benchmark.DATASET_SETS.keys()) == set(dsmod.DATASET_SETS.keys())


def test_config_bridge_build_base_config_matches_runner_helper():
    parser = benchmark.build_arg_parser()
    args = parser.parse_args(["--datasets", "synthetic_easy_dfshift"])
    spec = benchmark.BENCHMARK_DATASETS["synthetic_easy_dfshift"]

    cfg_bridge = build_base_config(args, spec, seed=11)
    cfg_runner = benchmark._build_base_config(args, spec, seed=11)

    assert cfg_bridge.random_seed == cfg_runner.random_seed
    assert tuple(cfg_bridge.enabled_methods) == tuple(cfg_runner.enabled_methods)
    assert cfg_bridge.dist_criterion == cfg_runner.dist_criterion
    assert cfg_bridge.df_stage_position == cfg_runner.df_stage_position

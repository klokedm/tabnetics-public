import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_df_fs_sota_benchmark as exp_benchmark
from tabnetics.benchmarks import runner as benchmark


@pytest.mark.parametrize("mode", ["tritrust", "uniform", "banzhaf", "shapley"])
def test_classifier_oracle_weighting_modes_are_accepted_by_benchmark_cli(mode):
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--seeds",
            "11",
            "--ablation-profile",
            "none",
            "--classifier-selection-mode",
            "mnpo_hybrid",
            "--classifier-oracle-weighting-mode",
            mode,
        ]
    )
    assert args.classifier_oracle_weighting_mode == mode


@pytest.mark.parametrize("mode", ["tritrust", "uniform", "banzhaf", "shapley"])
def test_classifier_oracle_weighting_modes_are_accepted_by_experiment_cli(mode):
    args = exp_benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--seeds",
            "11",
            "--ablation-profile",
            "none",
            "--classifier-selection-mode",
            "mnpo_hybrid",
            "--classifier-oracle-weighting-mode",
            mode,
        ]
    )
    assert args.classifier_oracle_weighting_mode == mode

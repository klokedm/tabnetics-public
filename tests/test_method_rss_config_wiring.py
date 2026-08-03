"""Public configuration and runner wiring for per-method FS RSS limits."""

from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.datasets import make_classification

from tabnetics.benchmarks import runner as benchmark
from tabnetics.feature_selection import FeatureSelector
from tabnetics.feature_selection.config import FeatureSelectorConfig
from tabnetics.pipeline.pipeline import (
    DFFSConfig,
    DistributionFeatureSelectionPipeline,
)
from tabnetics.validation.core import shard_runner


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), "invalid"])
def test_feature_selector_config_rejects_invalid_method_rss_limit(invalid: object) -> None:
    with pytest.raises(ValueError, match="method_max_rss_mb"):
        FeatureSelectorConfig(method_max_rss_mb=invalid)


def test_feature_selector_config_forwards_method_rss_limit() -> None:
    config = FeatureSelectorConfig(method_max_rss_mb="384.5")

    selector = FeatureSelector.from_config(config)

    assert config.method_max_rss_mb == pytest.approx(384.5)
    assert selector.method_max_rss_mb == pytest.approx(384.5)


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), "invalid"])
def test_pipeline_config_rejects_invalid_method_rss_limit(invalid: object) -> None:
    with pytest.raises(ValueError, match="fs_method_max_rss_mb"):
        DFFSConfig(fs_method_max_rss_mb=invalid)


def test_pipeline_forwards_method_rss_limit_through_direct_and_structured_paths() -> None:
    direct = DistributionFeatureSelectionPipeline(DFFSConfig(fs_method_max_rss_mb=768.0))
    direct_selector = direct._build_feature_selector(
        seed=11,
        enabled_methods=("mutual_information", "anova_f"),
    )

    structured = DistributionFeatureSelectionPipeline(
        DFFSConfig(
            fs_method_max_rss_mb=768.0,
            fs_config=FeatureSelectorConfig(method_max_rss_mb=64.0),
        )
    )
    structured_selector = structured._build_feature_selector(
        seed=11,
        enabled_methods=("mutual_information", "anova_f"),
    )

    assert direct_selector.method_max_rss_mb == pytest.approx(768.0)
    assert structured_selector.method_max_rss_mb == pytest.approx(768.0)


def test_pipeline_records_method_rss_limit_in_result_snapshot() -> None:
    X, y = make_classification(
        n_samples=60,
        n_features=16,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=19,
    )
    pipeline = DistributionFeatureSelectionPipeline(
        DFFSConfig(
            random_seed=19,
            fs_fraction=0.60,
            n_final_features=4,
            n_jobs=1,
            max_dist_features=0,
            apply_cdf_transform=False,
            use_rank_prefilter=False,
            screening_enabled=False,
            selection_strategy="legacy_voting",
            enabled_methods=("mutual_information", "anova_f"),
            fs_method_max_rss_mb=2048.0,
            model_candidates=("lr",),
        )
    )

    result = pipeline.run(X, y, dataset_name="rss-wiring", seed=19)

    assert result.config_snapshot["fs_method_max_rss_mb"] == pytest.approx(2048.0)


def test_benchmark_cli_builds_and_clones_method_rss_limit() -> None:
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--fs-method-max-rss-mb",
            "512",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_easy_dfshift"]

    config = benchmark._build_base_config(args, spec, seed=11)
    cloned = benchmark.clone_config(config)

    assert args.fs_method_max_rss_mb == pytest.approx(512.0)
    assert config.fs_method_max_rss_mb == pytest.approx(512.0)
    assert cloned.fs_method_max_rss_mb == pytest.approx(512.0)


def test_validation_shard_forwards_plan_and_environment_rss_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = {
        "job_id": "rss-wiring",
        "params": {
            "datasets": ["synthetic_easy_dfshift"],
            "seeds": [11],
            "fs_method_set": "strict_plus_mrmr",
            "fs_method_max_rss_mb": 640.0,
        },
    }

    from_plan = shard_runner._build_benchmark_cmd(
        job,
        out_dir=tmp_path,
        max_workers=1,
    )
    plan_value_index = from_plan.index("--fs-method-max-rss-mb") + 1
    assert float(from_plan[plan_value_index]) == pytest.approx(640.0)

    monkeypatch.setenv("FS_METHOD_MAX_RSS_MB_OVERRIDE", "896")
    from_environment = shard_runner._build_benchmark_cmd(
        job,
        out_dir=tmp_path,
        max_workers=1,
    )
    environment_value_index = from_environment.index("--fs-method-max-rss-mb") + 1
    assert float(from_environment[environment_value_index]) == pytest.approx(896.0)

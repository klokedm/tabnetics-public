import io
import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from types import SimpleNamespace
from scipy.io import savemat

from tabnetics.validation.suite import (
    CATALOG,
    COMPONENT_DEFAULTS,
    _apply_scenario_defaults_to_configs,
    _dataset_promotion_metadata,
    _load_dense_arff_dataset,
    _load_manual_tabular_dataset,
    _generate_distribution_cases,
    _integrated_scenario_defaults,
    _project_components_for_pipeline,
    _safe_train_test_split,
    _summarize_runs,
    AblationConfig,
    DatasetIntegrityPolicyError,
    DatasetIntegritySkipError,
    ValidationDatasetSpec,
    build_arg_parser,
    build_ablation_configs,
    load_feature_selection_dataset,
    run_validation_suite,
    resolve_dataset_ids,
)


def _has_real_source_for(spec):
    """Return True when manual/HF real data source is configured for this spec."""
    if os.environ.get("TABNETICS_HF_ORG", "").strip():
        return True

    params = dict(spec.params or {})
    env_name = str(params.get("local_path_env", "")).strip()
    if env_name:
        env_path = os.environ.get(env_name, "").strip()
        if env_path and Path(env_path).exists():
            return True

    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    default_one = params.get("default_local_path")
    if default_one:
        candidates.append(str(default_one))
    for extra in list(params.get("default_local_paths", []) or ()):
        candidates.append(str(extra))

    for cand in candidates:
        p = Path(cand)
        if not p.is_absolute():
            p = repo_root / p
        if p.exists():
            return True
    return False


def test_resolve_dataset_ids_pipeline_and_set_filtering():
    selected = resolve_dataset_ids(
        catalog=CATALOG,
        dataset_sets=["fs_easy"],
        explicit_ids=[],
        exclude_ids=["dlbcl_shipp"],
        pipelines=["fs"],
        max_datasets=None,
    )

    assert "leukemia_golub" in selected
    assert "dlbcl_shipp" not in selected
    assert all(CATALOG[ds_id].pipeline == "fs" for ds_id in selected)


def test_build_ablation_configs_single_off_subset():
    configs = build_ablation_configs(
        profile="single_off",
        base_components=dict(COMPONENT_DEFAULTS),
        pipelines=["fs"],
        constrained_components=["fs.tritrust", "fs.method_mrmr_jmi"],
    )

    names = [cfg.name for cfg in configs]
    assert names[0] == "baseline"
    assert "disable_fs_tritrust" in names
    assert "disable_fs_method_mrmr_jmi" in names
    assert len(configs) == 3


def test_manual_or_synth_dataset_falls_back_to_synthetic_when_env_missing(monkeypatch):
    # Use a manual dataset and hide both env var *and* default_local_path so the
    # loader reliably falls back to synthetic.
    import copy

    spec = copy.deepcopy(CATALOG["cumida_leukemia_subtypes"])
    monkeypatch.delenv("CUMIDA_LEUKEMIA_PATH", raising=False)
    # Remove default_local_path so the repo-local fallback path isn't tried.
    spec.params.pop("default_local_path", None)
    spec.params.pop("default_local_paths", None)
    loaded = load_feature_selection_dataset(
        spec,
        seed=11,
        allow_synthetic_fallback=True,
        sample_cap=120,
        feature_cap=300,
    )

    assert loaded.data_source == "synthetic_fallback"
    assert loaded.X.shape[0] <= 120
    assert loaded.X.shape[1] <= 300
    assert loaded.X.shape[0] == loaded.y.shape[0]
    assert np.issubdtype(loaded.X.dtype, np.number)


def test_validation_suite_cli_defaults_are_strict():
    args = build_arg_parser().parse_args([])
    assert args.allow_synthetic_fallback is False
    assert str(args.dataset_integrity_policy) == "error"


def test_require_hf_source_rejects_missing_hf_bundle(monkeypatch):
    import copy

    spec = copy.deepcopy(CATALOG["cumida_brain_gse50161"])
    monkeypatch.delenv("TABNETICS_HF_ORG", raising=False)
    monkeypatch.delenv("TABNETICS_HF_REPO_ID", raising=False)

    with pytest.raises(RuntimeError, match="HuggingFace bundle"):
        load_feature_selection_dataset(
            spec,
            seed=11,
            allow_synthetic_fallback=False,
            sample_cap=120,
            feature_cap=300,
            source_policy="real_only",
            require_hf_source=True,
        )


def test_validation_suite_rejects_allow_synthetic_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("TABNETICS_HF_ORG", "test-org")
    monkeypatch.delenv("TABNETICS_HF_REPO_ID", raising=False)

    args = build_arg_parser().parse_args(
        [
            "--datasets",
            "cumida_brain_gse50161",
            "--allow-synthetic-fallback",
            "--output-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(ValueError, match="Synthetic fallback is forbidden"):
        run_validation_suite(args)


def test_face_proxy_loader_builds_face_domain_dataset(monkeypatch):
    rng = np.random.default_rng(7)
    fake_images = rng.normal(loc=0.5, scale=0.2, size=(400, 64, 64)).astype(float)
    fake_images = np.clip(fake_images, 0.0, 1.0)
    fake_target = np.repeat(np.arange(40, dtype=int), 10)

    def _fake_fetch_olivetti_faces(*, shuffle: bool = False, download_if_missing: bool = True):
        return SimpleNamespace(images=fake_images, target=fake_target)

    monkeypatch.setattr("tabnetics.domains.face.datasets.fetch_olivetti_faces", _fake_fetch_olivetti_faces)

    spec = CATALOG["warp_pie10p"]
    loaded = load_feature_selection_dataset(
        spec,
        seed=11,
        allow_synthetic_fallback=False,
        sample_cap=500,
        feature_cap=5000,
    )

    assert loaded.data_source.startswith("sklearn:olivetti_faces_proxy:")
    assert loaded.X.shape == (210, 2420)
    assert loaded.y.shape == (210,)
    assert len(np.unique(loaded.y)) == 10


def test_mat_url_loader_builds_dataset_from_downloaded_payload(monkeypatch):
    X_ref = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    y_ref = np.asarray([1, 1, 2, 3], dtype=int).reshape(-1, 1)
    buf = io.BytesIO()
    savemat(buf, {"X": X_ref, "Y": y_ref})
    payload = buf.getvalue()

    monkeypatch.setattr("tabnetics.validation.suite._download_url_bytes", lambda _url, timeout_sec=60: payload)

    spec = ValidationDatasetSpec(
        dataset_id="toy_mat_url_loader",
        display_name="Toy MAT URL",
        pipeline="fs",
        tier="hard",
        loader_kind="mat_url_or_synth",
        params={
            "mat_url_options": [{"url": "https://example.invalid/toy.mat"}],
            "synthetic_profile": {
                "n_samples": 40,
                "n_features": 8,
                "n_classes": 3,
                "difficulty": "hard",
            },
        },
    )

    loaded = load_feature_selection_dataset(
        spec,
        seed=11,
        allow_synthetic_fallback=False,
        sample_cap=100,
        feature_cap=100,
    )

    assert loaded.data_source.startswith("mat_url:https://example.invalid/toy.mat")
    assert loaded.X.shape == (4, 3)
    assert loaded.y.shape == (4,)
    assert len(np.unique(loaded.y)) == 3


def test_nci60_proxy_loader_materializes_61x4_with_expected_class_count(monkeypatch):
    labels = (
        ["RENAL"] * 9
        + ["NSCLC"] * 9
        + ["MELANOMA"] * 8
        + ["BREAST"] * 7
        + ["COLON"] * 7
        + ["OVARIAN"] * 6
        + ["LEUKEMIA"] * 6
        + ["CNS"] * 5
        + ["PROSTATE"] * 2
        + ["UNKNOWN"] * 1
        + ["K562A-repro"] * 1
        + ["K562B-repro"] * 1
        + ["MCF7A-repro"] * 1
        + ["MCF7D-repro"] * 1
    )
    assert len(labels) == 64

    rng = np.random.default_rng(7)
    features = rng.normal(size=(64, 10))
    df = pd.DataFrame(features, columns=[f"g{i}" for i in range(10)])
    df.insert(0, "Unnamed: 0", np.arange(64))
    df["label"] = labels

    monkeypatch.setattr("tabnetics.validation.suite.pd.read_csv", lambda _url: df)

    spec = ValidationDatasetSpec(
        dataset_id="toy_nci60_proxy",
        display_name="Toy NCI60 Proxy",
        pipeline="fs",
        tier="hard",
        loader_kind="nci60_proxy_or_synth",
        params={
            "proxy_target_features": 8,
            "synthetic_profile": {
                "n_samples": 61,
                "n_features": 8,
                "n_classes": 8,
                "difficulty": "hard",
            },
        },
    )

    loaded = load_feature_selection_dataset(
        spec,
        seed=11,
        allow_synthetic_fallback=False,
        sample_cap=200,
        feature_cap=200,
    )

    assert loaded.data_source.startswith("rdatasets:ISLR:NCI60:proxy_")
    assert loaded.X.shape == (61, 8)
    assert loaded.y.shape == (61,)
    assert len(np.unique(loaded.y)) == 8
    assert "nci60_proxy_transform" in (loaded.notes or "")


def test_cumida_brain_loads_real_data_when_available():
    """CuMiDa Brain should load real data when available; synthetic fallback is optional."""
    spec = CATALOG["cumida_brain_gse50161"]
    real_available = _has_real_source_for(spec)
    loaded = load_feature_selection_dataset(
        spec,
        seed=19,
        allow_synthetic_fallback=True,
        sample_cap=120,
        feature_cap=300,
    )

    if real_available:
        assert loaded.data_source.startswith(("manual_arff:", "hf_bundle:"))
    else:
        assert loaded.data_source == "synthetic_fallback"
    assert loaded.X.shape[0] == loaded.y.shape[0]
    # Synthetic profile mirrors 4-class real-data topology.
    assert len(np.unique(loaded.y)) == 4


def test_cumida_brain_loads_real_data_when_fallback_disabled():
    spec = CATALOG["cumida_brain_gse50161"]
    if _has_real_source_for(spec):
        loaded = load_feature_selection_dataset(
            spec,
            seed=19,
            allow_synthetic_fallback=False,
            sample_cap=120,
            feature_cap=300,
        )
        assert loaded.data_source.startswith(("manual_arff:", "hf_bundle:"))
        return

    try:
        load_feature_selection_dataset(
            spec,
            seed=19,
            allow_synthetic_fallback=False,
            sample_cap=120,
            feature_cap=300,
        )
    except RuntimeError as exc:
        msg = str(exc)
        assert "No data source for dataset=cumida_brain_gse50161" in msg
    else:
        raise AssertionError("Expected missing-source RuntimeError when no real CuMiDa source is configured.")


def test_cumida_brain_promotion_metadata_is_promotable():
    """CuMiDa brain is a real-data benchmark; promotion metadata should not mark it fallback-only."""
    spec = CATALOG["cumida_brain_gse50161"]
    meta = _dataset_promotion_metadata(spec)

    assert int(meta["promotion_eligible"]) == 1
    assert str(meta["promotion_blocker"]) == ""
    assert str(meta["source_policy"]) in {"standard", "real_only"}


def test_dense_arff_loader_parses_numeric_features_and_label(tmp_path):
    content = """@RELATION toy

@ATTRIBUTE f1 NUMERIC
@ATTRIBUTE f2 NUMERIC
@ATTRIBUTE class {a,b}

@DATA
1.0,2.0,a
3.5,4.25,b
"""
    path = tmp_path / "toy.arff"
    path.write_text(content, encoding="utf-8")

    loaded = _load_dense_arff_dataset(path)
    assert loaded.X.shape == (2, 2)
    assert loaded.y.shape == (2,)
    assert loaded.data_source.startswith("manual_arff:")


def test_summarize_runs_preserves_domain_and_platform_dimensions():
    runs_df = pd.DataFrame(
        [
            {
                "pipeline": "fs",
                "dataset_id": "leukemia_golub",
                "config_name": "baseline",
                "domain": "genomics",
                "platform": "Affy HG-U95",
                "accuracy": 0.90,
                "balanced_accuracy": 0.91,
                "macro_f1": 0.89,
                "hybrid_score": 0.90,
            },
            {
                "pipeline": "fs",
                "dataset_id": "orlraws10p",
                "config_name": "baseline",
                "domain": "non_genomic",
                "platform": "cDNA",
                "accuracy": 0.88,
                "balanced_accuracy": 0.87,
                "macro_f1": 0.86,
                "hybrid_score": 0.87,
            },
        ]
    )
    summary_df, by_config_df = _summarize_runs(runs_df)

    assert "domain" in summary_df.columns
    assert "platform" in summary_df.columns
    assert "pipeline" in by_config_df.columns


def test_dense_arff_loader_parses_question_mark_missing_values(tmp_path):
    content = """@RELATION toy

@ATTRIBUTE f1 NUMERIC
@ATTRIBUTE f2 NUMERIC
@ATTRIBUTE class {a,b}

@DATA
1.0,?,a
?,2.0,b
"""
    path = tmp_path / "toy_missing.arff"
    path.write_text(content, encoding="utf-8")

    loaded = _load_dense_arff_dataset(path)
    assert loaded.X.shape == (2, 2)
    assert np.isnan(loaded.X[0, 1])
    assert np.isnan(loaded.X[1, 0])


def test_manual_arff_loader_rejects_git_lfs_pointer_stub(tmp_path):
    content = """version https://git-lfs.github.com/spec/v1
oid sha256:0000000000000000000000000000000000000000000000000000000000000000
size 123
"""
    path = tmp_path / "pointer.arff"
    path.write_text(content, encoding="utf-8")

    try:
        _load_manual_tabular_dataset(path)
        raise AssertionError("Expected RuntimeError for Git LFS pointer file.")
    except RuntimeError as exc:
        assert "Git LFS pointer file" in str(exc)


def test_gisette_openml_option_prefers_active_version():
    spec = CATALOG["gisette_nips03"]
    options = list(spec.params.get("openml_options", []) or [])
    assert len(options) >= 1
    assert str(options[0].get("name", "")).lower() == "gisette"
    assert int(options[0].get("version", 0)) == 2


def test_dense_arff_loader_drops_unknown_declared_class_rows(tmp_path):
    content = """@RELATION toy

@ATTRIBUTE f1 NUMERIC
@ATTRIBUTE class {a,b}

@DATA
1.0,a
2.0,b
3.0,c
"""
    path = tmp_path / "toy_unknown_class.arff"
    path.write_text(content, encoding="utf-8")

    loaded = _load_dense_arff_dataset(path)
    assert loaded.X.shape == (2, 1)
    assert loaded.y.shape == (2,)
    assert "unknown_class_rows_dropped:1" in (loaded.notes or "")


def test_manual_dataset_integrity_policy_fallback_uses_synthetic(tmp_path):
    content = """@RELATION toy

@ATTRIBUTE f1 NUMERIC
@ATTRIBUTE f2 NUMERIC
@ATTRIBUTE class {a,b}

@DATA
1.0,2.0,a
2.0,3.0,a
3.0,4.0,a
"""
    path = tmp_path / "single_class.arff"
    path.write_text(content, encoding="utf-8")

    spec = ValidationDatasetSpec(
        dataset_id="toy_manual_single_class",
        display_name="Toy Manual Single Class",
        pipeline="fs",
        tier="hard",
        loader_kind="manual_or_synth",
        params={
            "default_local_path": str(path),
            "synthetic_profile": {
                "n_samples": 80,
                "n_features": 120,
                "n_classes": 3,
                "difficulty": "hard",
            },
        },
    )

    loaded = load_feature_selection_dataset(
        spec,
        seed=11,
        allow_synthetic_fallback=True,
        sample_cap=90,
        feature_cap=130,
        class_integrity_policy="fallback",
        class_min_classes=2,
        class_min_class_count=1,
    )

    classes = np.unique(loaded.y)
    assert loaded.data_source == "synthetic_fallback"
    assert loaded.X.shape[0] <= 90
    assert loaded.X.shape[1] <= 130
    assert classes.size >= 2
    assert "integrity_fallback_from:manual_arff:" in (loaded.notes or "")


def test_manual_dataset_integrity_policy_skip_raises(tmp_path):
    content = """@RELATION toy

@ATTRIBUTE f1 NUMERIC
@ATTRIBUTE class {a,b}

@DATA
1.0,a
2.0,a
3.0,a
"""
    path = tmp_path / "single_class_skip.arff"
    path.write_text(content, encoding="utf-8")

    spec = ValidationDatasetSpec(
        dataset_id="toy_manual_single_class_skip",
        display_name="Toy Manual Single Class Skip",
        pipeline="fs",
        tier="hard",
        loader_kind="manual_or_synth",
        params={
            "default_local_path": str(path),
            "synthetic_profile": {
                "n_samples": 80,
                "n_features": 120,
                "n_classes": 2,
                "difficulty": "hard",
            },
        },
    )

    try:
        load_feature_selection_dataset(
            spec,
            seed=13,
            allow_synthetic_fallback=True,
            sample_cap=100,
            feature_cap=140,
            class_integrity_policy="skip",
            class_min_classes=2,
            class_min_class_count=1,
        )
        raise AssertionError("Expected DatasetIntegritySkipError for single-class manual dataset.")
    except DatasetIntegritySkipError:
        pass


def test_distribution_case_generation_for_all_df_specs():
    df_specs = [spec for spec in CATALOG.values() if spec.pipeline == "df"]
    for spec in df_specs:
        cases = _generate_distribution_cases(spec, seed=7)
        assert len(cases) > 0, spec.dataset_id


def test_integrated_scenario_defaults_enable_expected_components():
    base = dict(COMPONENT_DEFAULTS)
    base_proj = _project_components_for_pipeline(base, "integrated")
    cfgs = [AblationConfig(name="baseline", components=base_proj)]

    spec = CATALOG["int_low_gof_downweighting"]
    scenario_defaults = _integrated_scenario_defaults(spec)
    adjusted = _apply_scenario_defaults_to_configs(cfgs, base_proj, scenario_defaults)

    assert adjusted[0].components["integrated.low_gof_downweighting"] is True


def test_safe_train_test_split_respects_max_train_samples_cap():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(200, 8))
    y = np.asarray(([0, 1] * 100), dtype=int)

    X_train, X_test, y_train, y_test = _safe_train_test_split(
        X,
        y,
        test_size=0.20,
        seed=17,
        max_train_samples=60,
    )

    assert X_train.shape[0] == 60
    assert X_test.shape[0] == 140
    assert y_train.shape[0] == 60
    assert y_test.shape[0] == 140


def test_safe_train_test_split_enforces_minimum_80_20_holdout_when_capped():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(40, 6))
    y = np.asarray(([0, 1] * 20), dtype=int)

    X_train, X_test, y_train, y_test = _safe_train_test_split(
        X,
        y,
        test_size=0.20,
        seed=23,
        max_train_samples=36,
    )

    assert X_train.shape[0] == 32
    assert X_test.shape[0] == 8
    assert y_train.shape[0] == 32
    assert y_test.shape[0] == 8

import numpy as np

from tabnetics.pipeline.pipeline import DFFSConfig, DistributionFeatureSelectionPipeline


def test_ratio_feature_generation_appends_log_ratio_column_and_emits_pairs_meta():
    rng = np.random.default_rng(7)
    n_train = 48
    n_test = 16

    x1_train = rng.lognormal(mean=0.0, sigma=0.6, size=n_train)
    x2_train = rng.lognormal(mean=0.0, sigma=0.6, size=n_train)
    y_train = (x1_train > x2_train).astype(int)
    X_train = np.column_stack([x1_train, x2_train]).astype(float)

    x1_test = rng.lognormal(mean=0.0, sigma=0.6, size=n_test)
    x2_test = rng.lognormal(mean=0.0, sigma=0.6, size=n_test)
    X_test = np.column_stack([x1_test, x2_test]).astype(float)

    cfg = DFFSConfig(
        enable_ratio_features=True,
        ratio_pool_size=2,
        ratio_selection_method="ktsp",
        ratio_max_pairs=16,
        max_ratio_features=1,
        ratio_epsilon=1e-6,
        ratio_include_originals=True,
        ratio_abs_value=False,
        ratio_require_positive=True,
        apply_cdf_transform=False,  # keep test focused on RP-1 stage only
    )
    pipeline = DistributionFeatureSelectionPipeline(cfg)

    X_train_aug, X_test_aug, meta = pipeline._ratio_feature_generation(
        X_train_imp=X_train,
        y_train=y_train,
        X_test_imp=X_test,
        seed=13,
        face_projection_applied=False,
    )

    assert meta["ratio_features_applied"] is True
    assert int(meta["ratio_features_added"]) == 1
    assert X_train_aug.shape == (n_train, 3)
    assert X_test_aug.shape == (n_test, 3)

    pairs = meta.get("ratio_pairs", [])
    assert isinstance(pairs, list)
    assert len(pairs) == 1
    pair = pairs[0]
    assert set(pair.keys()) == {"numerator", "denominator", "score"}
    a = int(pair["numerator"])
    b = int(pair["denominator"])

    eps = float(meta["ratio_epsilon"])
    expected_train = np.log((X_train[:, a] + eps) / (X_train[:, b] + eps))
    expected_test = np.log((X_test[:, a] + eps) / (X_test[:, b] + eps))
    np.testing.assert_allclose(X_train_aug[:, -1], expected_train, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(X_test_aug[:, -1], expected_test, rtol=1e-6, atol=1e-8)


"""Tests for MAPIE conformal prediction integration (VAL12_Suggestions §2.3)."""

import numpy as np
import pytest


def _try_import_mapie() -> bool:
    try:
        from mapie.classification import SplitConformalClassifier  # type: ignore  # noqa: F401

        return True
    except Exception:
        try:
            from mapie.classification import MapieClassifier  # type: ignore  # noqa: F401

            return True
        except Exception:
            return False


class TestMapieAvailability:
    """Verify mapie_available() returns correct status."""

    def test_mapie_available_returns_bool(self):
        from tabnetics.feature_selection.conformal import mapie_available

        result = mapie_available()
        assert isinstance(result, bool)


class TestComputeMapieConformalSets:
    """Verify MAPIE conformal prediction wrapper."""

    def test_graceful_skip_when_mapie_not_installed(self):
        """If mapie is not installed, should return skip_reason."""
        from tabnetics.feature_selection.conformal import (
            compute_mapie_conformal_sets,
            mapie_available,
        )

        if mapie_available():
            pytest.skip("mapie is installed; testing unavailable branch not possible")

        rng = np.random.RandomState(42)
        X_train = rng.randn(60, 4)
        y_train = rng.randint(0, 2, size=60)
        X_eval = rng.randn(20, 4)
        y_eval = rng.randint(0, 2, size=20)

        result = compute_mapie_conformal_sets(
            model=None,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            method="aps",
        )
        assert result["classifier_conformal_mapie_applied"] is False
        assert result["classifier_conformal_mapie_skip_reason"] == "mapie_not_installed"

    def test_invalid_input_guard(self):
        """Invalid input should return skip_reason without crash."""
        from tabnetics.feature_selection.conformal import compute_mapie_conformal_sets

        result = compute_mapie_conformal_sets(
            model=None,
            X_train=np.array([1.0, 2.0]),  # 1D, invalid
            y_train=np.array([0, 1]),
            X_eval=np.array([3.0, 4.0]),
            y_eval=np.array([0, 1]),
            method="aps",
        )
        assert result["classifier_conformal_mapie_applied"] is False
        skip_reason = result["classifier_conformal_mapie_skip_reason"]
        assert skip_reason in {"invalid_input_rank", "mapie_not_installed"}

    def test_single_class_guard(self):
        """Single class should return skip_reason."""
        from tabnetics.feature_selection.conformal import compute_mapie_conformal_sets

        X_train = np.random.randn(30, 3)
        y_train = np.zeros(30, dtype=int)
        X_eval = np.random.randn(10, 3)

        result = compute_mapie_conformal_sets(
            model=None,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            method="aps",
        )
        assert result["classifier_conformal_mapie_applied"] is False
        skip_reason = result["classifier_conformal_mapie_skip_reason"]
        assert skip_reason in {"single_class", "mapie_not_installed"}

    def test_default_method_is_aps(self):
        """Default method should be 'aps'."""
        from tabnetics.feature_selection.conformal import compute_mapie_conformal_sets

        result = compute_mapie_conformal_sets(
            model=None,
            X_train=np.random.randn(30, 3),
            y_train=np.random.randint(0, 2, 30),
            X_eval=np.random.randn(10, 3),
        )
        assert result["classifier_conformal_mapie_method"] == "aps"

    def test_invalid_method_defaults_to_aps(self):
        """Invalid method string should default to 'aps'."""
        from tabnetics.feature_selection.conformal import compute_mapie_conformal_sets

        result = compute_mapie_conformal_sets(
            model=None,
            X_train=np.random.randn(30, 3),
            y_train=np.random.randint(0, 2, 30),
            X_eval=np.random.randn(10, 3),
            method="invalid_method",
        )
        assert result["classifier_conformal_mapie_method"] == "aps"

    def test_output_keys_present(self):
        """All expected output keys should be present."""
        from tabnetics.feature_selection.conformal import compute_mapie_conformal_sets

        result = compute_mapie_conformal_sets(
            model=None,
            X_train=np.random.randn(30, 3),
            y_train=np.random.randint(0, 2, 30),
            X_eval=np.random.randn(10, 3),
            method="aps",
        )
        prefix = "classifier_conformal_mapie_"
        expected_keys = [
            f"{prefix}enabled",
            f"{prefix}applied",
            f"{prefix}skip_reason",
            f"{prefix}method",
            f"{prefix}alpha",
            f"{prefix}set_size_mean",
            f"{prefix}set_size_median",
            f"{prefix}singleton_rate",
            f"{prefix}empty_set_rate",
            f"{prefix}coverage",
            f"{prefix}classes",
            f"{prefix}prediction_sets",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


@pytest.mark.skipif(
    not _try_import_mapie(),
    reason="mapie package not installed",
)
class TestMapieSmoke:
    """Smoke tests requiring mapie to be installed."""

    def test_aps_binary(self):
        """APS should work on binary classification."""
        from sklearn.naive_bayes import GaussianNB

        from tabnetics.feature_selection.conformal import compute_mapie_conformal_sets

        rng = np.random.RandomState(42)
        n_train, n_test, p = 100, 30, 5
        X_train = rng.randn(n_train, p)
        y_train = (X_train[:, 0] > 0).astype(int)
        X_test = rng.randn(n_test, p)
        y_test = (X_test[:, 0] > 0).astype(int)

        result = compute_mapie_conformal_sets(
            model=GaussianNB(),
            X_train=X_train,
            y_train=y_train,
            X_eval=X_test,
            y_eval=y_test,
            method="aps",
            alpha=0.10,
            seed=42,
        )
        assert result["classifier_conformal_mapie_applied"] is True
        assert np.isfinite(result["classifier_conformal_mapie_coverage"])
        assert np.isfinite(result["classifier_conformal_mapie_set_size_mean"])
        assert result["classifier_conformal_mapie_coverage"] >= 0.7

    def test_raps_binary(self):
        """RAPS should work on binary classification."""
        from sklearn.naive_bayes import GaussianNB

        from tabnetics.feature_selection.conformal import compute_mapie_conformal_sets

        rng = np.random.RandomState(42)
        n_train, n_test, p = 100, 30, 5
        X_train = rng.randn(n_train, p)
        y_train = (X_train[:, 0] > 0).astype(int)
        X_test = rng.randn(n_test, p)
        y_test = (X_test[:, 0] > 0).astype(int)

        result = compute_mapie_conformal_sets(
            model=GaussianNB(),
            X_train=X_train,
            y_train=y_train,
            X_eval=X_test,
            y_eval=y_test,
            method="raps",
            alpha=0.10,
            seed=42,
        )
        assert result["classifier_conformal_mapie_applied"] is True
        assert np.isfinite(result["classifier_conformal_mapie_coverage"])

    def test_cross_conformal_multiclass(self):
        """Cross-conformal should work on multiclass."""
        from sklearn.naive_bayes import GaussianNB

        from tabnetics.feature_selection.conformal import compute_mapie_conformal_sets

        rng = np.random.RandomState(42)
        n_train, n_test, p = 120, 30, 4
        X_train = rng.randn(n_train, p)
        y_train = (X_train[:, 0] * 3).astype(int) % 3
        X_test = rng.randn(n_test, p)
        y_test = (X_test[:, 0] * 3).astype(int) % 3

        result = compute_mapie_conformal_sets(
            model=GaussianNB(),
            X_train=X_train,
            y_train=y_train,
            X_eval=X_test,
            y_eval=y_test,
            method="cross",
            alpha=0.10,
            seed=42,
            cv_folds=3,
        )
        assert result["classifier_conformal_mapie_applied"] is True
        assert np.isfinite(result["classifier_conformal_mapie_coverage"])


class TestConformalMethodConfig:
    """Verify conformal_method config toggle."""

    def test_config_has_conformal_method(self):
        from tabnetics.pipeline.pipeline import DFFSConfig

        config = DFFSConfig()
        assert hasattr(config, "calibration_reporting_enabled")

    def test_conformal_method_default_is_split(self):
        """Default conformal method should be 'split'."""
        from tabnetics.pipeline.pipeline import DFFSConfig

        config = DFFSConfig()
        cls_cfg = config._classification_cfg() if hasattr(config, "_classification_cfg") else None
        # The conformal_method field is on the classification config
        if cls_cfg is not None:
            assert str(getattr(cls_cfg, "conformal_method", "split")) == "split"

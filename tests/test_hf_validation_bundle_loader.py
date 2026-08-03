"""
Tests for HF validation bundle loader.

Tests the new _load_hf_parquet_dataset() that uses the single HF validation
bundle repo (config-per-dataset_id) instead of per-dataset repos.

T-VIH-009 / AF-010 fix.
"""
import os
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from tabnetics.validation.suite import _load_hf_parquet_dataset


class TestHFBundleLoader:
    """Test the HF validation bundle loader with fail-fast behavior (no fallback)."""

    def test_load_success_with_features_and_label(self):
        """Test successful loading from HF bundle with features/label columns."""
        # Mock the datasets.load_dataset call
        mock_dataset = MagicMock()
        mock_dataset.features = {
            "features": Mock(),
            "label": Mock(),
        }
        # Simulate HF dataset with list[float32] features and int64 labels
        mock_dataset.__getitem__.side_effect = lambda key: {
            "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "label": [0, 1],
        }[key]
        
        with patch("datasets.load_dataset", return_value=mock_dataset):
            result = _load_hf_parquet_dataset("test_dataset", hf_org="test_org")
        
        assert result.X.shape == (2, 3)
        assert result.y.shape == (2,)
        assert result.data_source == "hf_bundle:test_org/tabnetics-validation/test_dataset"
        assert "loaded_from_hf_bundle:config=test_dataset" in result.notes

    def test_load_raises_when_dataset_config_missing(self):
        """Test that loader raises (no fallback) when dataset/config is missing."""
        with patch(
            "datasets.load_dataset",
            side_effect=Exception("Config 'nonexistent' not found"),
        ):
            with pytest.raises(RuntimeError, match="Failed to load dataset 'nonexistent'"):
                _load_hf_parquet_dataset("nonexistent", hf_org="test_org")

    def test_load_raises_when_missing_features_column(self):
        """Test that loader raises when 'features' column is missing."""
        mock_dataset = Mock()
        mock_dataset.features = {"label": Mock()}  # Missing 'features'
        
        with patch("datasets.load_dataset", return_value=mock_dataset):
            with pytest.raises(RuntimeError, match="missing expected columns"):
                _load_hf_parquet_dataset("bad_dataset", hf_org="test_org")

    def test_load_raises_when_missing_label_column(self):
        """Test that loader raises when 'label' column is missing."""
        mock_dataset = Mock()
        mock_dataset.features = {"features": Mock()}  # Missing 'label'
        
        with patch("datasets.load_dataset", return_value=mock_dataset):
            with pytest.raises(RuntimeError, match="missing expected columns"):
                _load_hf_parquet_dataset("bad_dataset", hf_org="test_org")

    def test_load_raises_when_datasets_not_installed(self):
        """Test that loader raises when datasets library is not installed."""
        # Temporarily replace the import with an ImportError
        import sys
        orig_datasets = sys.modules.get("datasets")
        sys.modules["datasets"] = None
        
        try:
            # Force reimport to trigger the ImportError
            import importlib
            import tabnetics.validation.suite as vs
            importlib.reload(vs)
            
            with pytest.raises(RuntimeError, match="datasets library not installed"):
                vs._load_hf_parquet_dataset("test_dataset", hf_org="test_org")
        finally:
            # Restore the original module
            if orig_datasets is not None:
                sys.modules["datasets"] = orig_datasets
            else:
                sys.modules.pop("datasets", None)
            # Reload to restore original state
            import importlib
            import tabnetics.validation.suite as vs
            importlib.reload(vs)

    def test_custom_bundle_repo_name(self):
        """Test that custom bundle_repo_name parameter is used."""
        mock_dataset = MagicMock()
        mock_dataset.features = {"features": Mock(), "label": Mock()}
        mock_dataset.__getitem__.side_effect = lambda key: {
            "features": [[1.0, 2.0]],
            "label": [0],
        }[key]
        
        with patch("datasets.load_dataset", return_value=mock_dataset) as mock_load:
            result = _load_hf_parquet_dataset(
                "test_dataset",
                hf_org="test_org",
                bundle_repo_name="custom-bundle-name",
            )
            
            # Verify the correct repo_id was used
            mock_load.assert_called_once_with(
                "test_org/custom-bundle-name",
                name="test_dataset",
                split="train",
            )
            assert "test_org/custom-bundle-name" in result.data_source

    def test_label_encoding_applied(self):
        """Test that labels are passed through (existing _safe_label_encode behavior for integers)."""
        mock_dataset = MagicMock()
        mock_dataset.features = {"features": Mock(), "label": Mock()}
        # Integer labels are passed through by _safe_label_encode (existing behavior)
        mock_dataset.__getitem__.side_effect = lambda key: {
            "features": [[1.0], [2.0], [3.0], [4.0]],
            "label": [0, 1, 0, 1],
        }[key]
        
        with patch("datasets.load_dataset", return_value=mock_dataset):
            result = _load_hf_parquet_dataset("test_dataset", hf_org="test_org")
        
        # Labels should be preserved as-is for integer dtypes
        assert set(result.y) == {0, 1}
        assert result.y[0] == result.y[2]
        assert result.y[1] == result.y[3]


class TestHFBundleLoaderIntegration:
    """Integration tests that attempt to load from the actual HF bundle (if configured)."""

    @pytest.mark.skipif(
        not os.environ.get("TABNETICS_HF_ORG") or not os.environ.get("HF_TOKEN"),
        reason="TABNETICS_HF_ORG and HF_TOKEN must be set for integration tests",
    )
    def test_load_real_dataset_from_bundle(self):
        """Integration test: load a real dataset from the actual HF bundle."""
        hf_org = os.environ["TABNETICS_HF_ORG"]
        # Use a dataset that should exist in the bundle
        result = _load_hf_parquet_dataset("leukemia_golub", hf_org=hf_org)
        
        assert result.X.shape[0] > 0  # Has samples
        assert result.X.shape[1] > 0  # Has features
        assert result.y.shape[0] == result.X.shape[0]  # Labels match samples
        assert result.data_source.startswith("hf_bundle:")
        assert "leukemia_golub" in result.data_source

    @pytest.mark.skipif(
        not os.environ.get("TABNETICS_HF_ORG") or not os.environ.get("HF_TOKEN"),
        reason="TABNETICS_HF_ORG and HF_TOKEN must be set for integration tests",
    )
    def test_load_nonexistent_dataset_raises(self):
        """Integration test: verify that nonexistent dataset raises RuntimeError."""
        hf_org = os.environ["TABNETICS_HF_ORG"]
        
        with pytest.raises(RuntimeError, match="Failed to load dataset"):
            _load_hf_parquet_dataset("nonexistent_dataset_12345", hf_org=hf_org)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

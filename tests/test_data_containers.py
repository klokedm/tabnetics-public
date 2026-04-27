"""Tests for OmicsDataset container (VAL12_Suggestions §3.1)."""

import numpy as np
import pandas as pd
import pytest

from tabnetics.datasets.containers import OmicsDataset


class TestConstruction:
    """Verify basic construction."""

    def test_basic_construction(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        y = rng.randint(0, 2, 20)
        ds = OmicsDataset(X=X, y=y)
        assert ds.n_samples == 20
        assert ds.n_features == 10
        assert ds.n_classes == 2
        assert ds.assay_type == "unknown"

    def test_with_metadata(self):
        rng = np.random.RandomState(42)
        X = rng.randn(15, 5)
        y = rng.randint(0, 3, 15)
        sample_meta = pd.DataFrame({"batch": [0] * 7 + [1] * 8})
        feature_meta = pd.DataFrame({"gene": [f"G{i}" for i in range(5)]})
        ds = OmicsDataset(
            X=X,
            y=y,
            sample_meta=sample_meta,
            feature_meta=feature_meta,
            assay_type="microarray",
            batch_key="batch",
        )
        assert ds.n_samples == 15
        assert ds.n_features == 5
        assert ds.assay_type == "microarray"
        assert ds.batch_key == "batch"
        bl = ds.batch_labels
        assert bl is not None
        assert bl.size == 15

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            OmicsDataset(X=np.array([1, 2, 3]), y=np.array([0, 1, 0]))

    def test_y_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="y length"):
            OmicsDataset(X=np.random.randn(10, 5), y=np.array([0, 1]))


class TestProperties:
    """Verify computed properties."""

    def test_class_counts(self):
        X = np.random.randn(12, 4)
        y = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        ds = OmicsDataset(X=X, y=y)
        counts = ds.class_counts
        assert counts.size == 3
        np.testing.assert_array_equal(np.sort(counts), [3, 4, 5])

    def test_batch_labels_none_without_meta(self):
        ds = OmicsDataset(X=np.random.randn(5, 3), y=np.zeros(5))
        assert ds.batch_labels is None

    def test_batch_labels_none_without_key(self):
        sample_meta = pd.DataFrame({"batch": [0, 1, 0, 1, 0]})
        ds = OmicsDataset(
            X=np.random.randn(5, 3),
            y=np.zeros(5),
            sample_meta=sample_meta,
            batch_key=None,
        )
        assert ds.batch_labels is None


class TestSubsetting:
    """Verify feature and sample subsetting."""

    def _make_dataset(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        y = rng.randint(0, 2, 20)
        sample_meta = pd.DataFrame({
            "sample_id": [f"S{i}" for i in range(20)],
            "batch": rng.randint(0, 3, 20),
        })
        feature_meta = pd.DataFrame({
            "gene": [f"G{i}" for i in range(10)],
        })
        return OmicsDataset(
            X=X,
            y=y,
            sample_meta=sample_meta,
            feature_meta=feature_meta,
            assay_type="rnaseq",
            batch_key="batch",
        )

    def test_subset_features(self):
        ds = self._make_dataset()
        sub = ds.subset_features([0, 3, 7])
        assert sub.n_samples == 20
        assert sub.n_features == 3
        np.testing.assert_array_equal(sub.X, ds.X[:, [0, 3, 7]])
        assert sub.feature_meta is not None
        assert sub.feature_meta.shape[0] == 3

    def test_subset_samples(self):
        ds = self._make_dataset()
        sub = ds.subset_samples([0, 5, 10, 15])
        assert sub.n_samples == 4
        assert sub.n_features == 10
        np.testing.assert_array_equal(sub.y, ds.y[[0, 5, 10, 15]])
        assert sub.sample_meta is not None
        assert sub.sample_meta.shape[0] == 4

    def test_subset_preserves_assay_type(self):
        ds = self._make_dataset()
        sub = ds.subset_features([1, 2])
        assert sub.assay_type == "rnaseq"

    def test_modality_membership_remapping(self):
        X = np.random.randn(10, 8)
        y = np.zeros(10)
        ds = OmicsDataset(
            X=X,
            y=y,
            modality_membership={"mrna": [0, 1, 2, 3], "protein": [4, 5, 6, 7]},
        )
        sub = ds.subset_features([1, 2, 5, 6])
        assert sub.modality_membership is not None
        assert "mrna" in sub.modality_membership
        assert "protein" in sub.modality_membership
        # Original indices [1,2] map to new positions [0,1]
        assert sub.modality_membership["mrna"] == [0, 1]
        # Original indices [5,6] map to new positions [2,3]
        assert sub.modality_membership["protein"] == [2, 3]


class TestTrainTestSplit:
    """Verify stratified train/test split."""

    def test_split_sizes(self):
        X = np.random.randn(100, 5)
        y = np.array([0] * 50 + [1] * 50)
        ds = OmicsDataset(X=X, y=y)
        train, test = ds.train_test_split(test_size=0.3, random_state=42)
        assert train.n_samples == 70
        assert test.n_samples == 30
        assert train.n_features == 5
        assert test.n_features == 5

    def test_split_stratified(self):
        X = np.random.randn(100, 5)
        y = np.array([0] * 50 + [1] * 50)
        ds = OmicsDataset(X=X, y=y)
        _, test = ds.train_test_split(test_size=0.2, random_state=42)
        # Stratified split should preserve class balance ± 1
        counts = test.class_counts
        assert abs(counts[0] - counts[1]) <= 2

    def test_split_deterministic(self):
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, 50)
        ds = OmicsDataset(X=X, y=y)
        t1, _ = ds.train_test_split(test_size=0.2, random_state=42)
        t2, _ = ds.train_test_split(test_size=0.2, random_state=42)
        np.testing.assert_array_equal(t1.X, t2.X)
        np.testing.assert_array_equal(t1.y, t2.y)


class TestConversion:
    """Verify to_Xy and from_Xy round-trip."""

    def test_to_xy(self):
        X = np.random.randn(10, 5)
        y = np.arange(10)
        ds = OmicsDataset(X=X, y=y)
        Xo, yo = ds.to_Xy()
        np.testing.assert_array_equal(Xo, X)
        np.testing.assert_array_equal(yo, y)
        # Should be copies, not views
        Xo[0, 0] = 999.0
        assert ds.X[0, 0] != 999.0

    def test_from_xy_basic(self):
        X = np.random.randn(10, 5)
        y = np.arange(10)
        ds = OmicsDataset.from_Xy(X, y, assay_type="proteomics")
        assert ds.n_samples == 10
        assert ds.n_features == 5
        assert ds.assay_type == "proteomics"

    def test_from_xy_with_batch(self):
        X = np.random.randn(10, 5)
        y = np.zeros(10)
        batch = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        ds = OmicsDataset.from_Xy(X, y, batch_labels=batch)
        assert ds.batch_key == "batch"
        bl = ds.batch_labels
        assert bl is not None
        np.testing.assert_array_equal(bl, batch)

    def test_from_xy_with_feature_names(self):
        X = np.random.randn(10, 3)
        y = np.zeros(10)
        names = ["BRCA1", "TP53", "EGFR"]
        ds = OmicsDataset.from_Xy(X, y, feature_names=names)
        assert ds.feature_meta is not None
        assert list(ds.feature_meta["feature_name"]) == names

    def test_round_trip(self):
        X = np.random.randn(20, 8)
        y = np.random.randint(0, 3, 20)
        ds = OmicsDataset.from_Xy(X, y, assay_type="microarray")
        Xo, yo = ds.to_Xy()
        ds2 = OmicsDataset.from_Xy(Xo, yo, assay_type="microarray")
        np.testing.assert_array_equal(ds.X, ds2.X)
        np.testing.assert_array_equal(ds.y, ds2.y)

    def test_repr(self):
        ds = OmicsDataset(X=np.random.randn(5, 3), y=np.zeros(5), assay_type="rnaseq")
        r = repr(ds)
        assert "n_samples=5" in r
        assert "n_features=3" in r
        assert "rnaseq" in r

"""Tests for group-aware feature selection (VAL12_Suggestions §3.3)."""

import numpy as np
import pytest

from tabnetics.feature_selection import FeatureSelector
from tabnetics.feature_selection.methods import group_fs as group_fs_module
from tabnetics.feature_selection.methods.group_fs import (
    discover_feature_groups,
    discover_pathway_feature_groups,
    group_sparse_lasso_selection,
    pathway_group_sparse_lasso_selection,
)


class TestDiscoverFeatureGroups:
    """Verify automatic group discovery."""

    def test_correlated_features_grouped(self):
        """Features in the same block should land in the same group."""
        rng = np.random.RandomState(42)
        n, p = 100, 12
        # Create 3 blocks of 4 correlated features each.
        base = rng.randn(n, 3)
        X = np.zeros((n, p), dtype=float)
        for block in range(3):
            for j in range(4):
                X[:, block * 4 + j] = base[:, block] + rng.randn(n) * 0.1
        groups = discover_feature_groups(X, distance_threshold=0.5)
        assert groups.shape == (p,)
        # Features 0-3 should share a group, 4-7 another, 8-11 another.
        assert len(set(groups[0:4].tolist())) == 1
        assert len(set(groups[4:8].tolist())) == 1
        assert len(set(groups[8:12].tolist())) == 1
        # The three groups should be distinct.
        assert groups[0] != groups[4]
        assert groups[4] != groups[8]

    def test_independent_features_separate_groups(self):
        """Independent features should get separate groups."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        groups = discover_feature_groups(X, distance_threshold=0.3)
        # With independent features, expect many groups.
        assert len(set(groups.tolist())) >= 5

    def test_single_feature(self):
        """Single feature should get group 0."""
        X = np.random.randn(20, 1)
        groups = discover_feature_groups(X)
        assert groups.shape == (1,)

    def test_max_group_size_enforced(self):
        """Large groups should be split when max_group_size is small."""
        rng = np.random.RandomState(42)
        n = 50
        base = rng.randn(n)
        X = np.column_stack([base + rng.randn(n) * 0.05 for _ in range(20)])
        groups = discover_feature_groups(X, distance_threshold=0.5, max_group_size=5)
        # All are correlated → originally 1 group, but split at size 5.
        for g in range(int(groups.max()) + 1):
            count = int(np.sum(groups == g))
            assert count <= 5, f"Group {g} has {count} members, exceeds max_group_size=5"


class TestGroupSparseLassoSelection:
    """Verify group sparse lasso feature selection."""

    def _make_block_data(self, seed=42):
        """Create data with 3 informative feature blocks and 1 noise block."""
        rng = np.random.RandomState(seed)
        n = 100
        # Block 1: features 0-3 (informative)
        base1 = rng.randn(n)
        X_block1 = np.column_stack([base1 + rng.randn(n) * 0.1 for _ in range(4)])
        # Block 2: features 4-7 (informative, different signal)
        base2 = rng.randn(n)
        X_block2 = np.column_stack([base2 + rng.randn(n) * 0.1 for _ in range(4)])
        # Block 3: features 8-11 (noise)
        X_block3 = rng.randn(n, 4) * 0.1
        # Block 4: features 12-15 (noise)
        X_block4 = rng.randn(n, 4) * 0.1
        X = np.hstack([X_block1, X_block2, X_block3, X_block4])
        y = (base1 + base2 > 0).astype(int)
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=int)
        return X, y, groups

    def test_returns_correct_format(self):
        X, y, groups = self._make_block_data()
        results, all_scores = group_sparse_lasso_selection(
            X, y, n_target_features=8, groups=groups, alpha=0.1,
        )
        assert "selected_indices" in results
        assert "scores" in results
        assert "all_scores" in results
        assert "n_groups_total" in results
        assert "n_groups_selected" in results
        assert isinstance(all_scores, dict)
        assert len(all_scores) == X.shape[1]

    def test_selects_informative_features(self):
        """Should prefer informative blocks over noise blocks."""
        X, y, groups = self._make_block_data()
        results, _ = group_sparse_lasso_selection(
            X, y, n_target_features=8, groups=groups, alpha=0.05,
        )
        selected = set(results["selected_indices"].tolist())
        informative = set(range(8))
        noise = set(range(8, 16))
        # Most selected should be from informative features.
        n_informative = len(selected & informative)
        n_noise = len(selected & noise)
        assert n_informative >= n_noise, (
            f"Expected more informative ({n_informative}) than noise ({n_noise})"
        )

    def test_respects_n_target(self):
        X, y, groups = self._make_block_data()
        for n_target in [4, 8, 12]:
            results, _ = group_sparse_lasso_selection(
                X, y, n_target_features=n_target, groups=groups,
            )
            assert results["selected_indices"].size == n_target

    def test_auto_group_discovery(self):
        """When groups=None, should discover groups automatically."""
        X, y, _ = self._make_block_data()
        results, _ = group_sparse_lasso_selection(
            X, y, n_target_features=8, groups=None, alpha=0.1,
        )
        assert results["n_groups_total"] >= 1
        assert "group_labels" in results
        assert len(results["group_labels"]) == X.shape[1]

    def test_multiclass_support(self):
        """Should work with more than 2 classes."""
        rng = np.random.RandomState(42)
        X = rng.randn(120, 10)
        y = np.repeat([0, 1, 2], 40)
        X[:40, :3] += 2.0
        X[40:80, 3:6] += 2.0
        X[80:, 6:9] += 2.0
        results, all_scores = group_sparse_lasso_selection(
            X, y, n_target_features=6, alpha=0.05,
        )
        assert results["selected_indices"].size == 6
        assert len(all_scores) == 10

    def test_deterministic(self):
        """Same seed → same results."""
        X, y, groups = self._make_block_data()
        r1, _ = group_sparse_lasso_selection(
            X, y, 8, groups=groups, random_state=42,
        )
        r2, _ = group_sparse_lasso_selection(
            X, y, 8, groups=groups, random_state=42,
        )
        np.testing.assert_array_equal(r1["selected_indices"], r2["selected_indices"])

    def test_group_metadata_in_results(self):
        X, y, groups = self._make_block_data()
        results, _ = group_sparse_lasso_selection(
            X, y, 8, groups=groups,
        )
        assert results["n_groups_total"] == 4
        assert results["n_groups_selected"] >= 1
        assert results["n_groups_selected"] <= 4

    def test_high_alpha_selects_fewer_nonzero(self):
        """Higher alpha should produce sparser importance scores."""
        X, y, groups = self._make_block_data()
        _, scores_low = group_sparse_lasso_selection(
            X, y, 8, groups=groups, alpha=0.01,
        )
        _, scores_high = group_sparse_lasso_selection(
            X, y, 8, groups=groups, alpha=1.0,
        )
        nonzero_low = sum(1 for v in scores_low.values() if abs(v) > 1e-8)
        nonzero_high = sum(1 for v in scores_high.values() if abs(v) > 1e-8)
        assert nonzero_high <= nonzero_low


class TestPathwayGroupSparseLassoSelection:
    """Verify training-data-derived signed pathway-proxy grouping."""

    def _make_signed_proxy_data(self, seed=7):
        rng = np.random.RandomState(seed)
        n = 80
        y = np.repeat([0, 1], n // 2)
        base_a = rng.randn(n)
        base_b = rng.randn(n)
        X = np.column_stack(
            [
                base_a + rng.randn(n) * 0.03,
                base_a + rng.randn(n) * 0.03,
                -base_a + rng.randn(n) * 0.03,
                base_b + rng.randn(n) * 0.03,
                base_b + rng.randn(n) * 0.03,
                rng.randn(n),
            ]
        )
        return X, y

    def test_signed_pathway_groups_differ_from_absolute_correlation_groups(self):
        X, y = self._make_signed_proxy_data()

        absolute_groups = discover_feature_groups(X, distance_threshold=0.2)
        pathway_groups = discover_pathway_feature_groups(
            X,
            y,
            n_groups=3,
            max_group_size=3,
            random_state=11,
        )

        assert pathway_groups.shape == absolute_groups.shape == (X.shape[1],)
        assert absolute_groups[0] == absolute_groups[2]
        assert pathway_groups[0] == pathway_groups[1]
        assert pathway_groups[0] != pathway_groups[2]

    def test_pathway_group_discovery_is_deterministic_and_enforces_group_size(self):
        X, y = self._make_signed_proxy_data()

        first = discover_pathway_feature_groups(
            X,
            y,
            n_groups=2,
            max_group_size=2,
            random_state=13,
        )
        second = discover_pathway_feature_groups(
            X,
            y,
            n_groups=2,
            max_group_size=2,
            random_state=13,
        )

        np.testing.assert_array_equal(first, second)
        assert max(int(np.sum(first == g)) for g in set(first.tolist())) <= 2

    def test_wide_pathway_group_discovery_avoids_all_pairs_graph(self, monkeypatch):
        rng = np.random.RandomState(23)
        X = rng.randn(36, 620)
        y = np.repeat([0, 1, 2], 12)

        def _fail_all_pairs(*args, **kwargs):  # pragma: no cover - regression guard
            raise AssertionError("wide pathway grouping must avoid all-pairs graph work")

        monkeypatch.setattr(group_fs_module.np, "corrcoef", _fail_all_pairs)
        monkeypatch.setattr(group_fs_module, "linkage", _fail_all_pairs)

        first = discover_pathway_feature_groups(
            X,
            y,
            n_groups=12,
            max_group_size=20,
            random_state=19,
        )
        second = discover_pathway_feature_groups(
            X,
            y,
            n_groups=12,
            max_group_size=20,
            random_state=19,
        )

        assert first.shape == (X.shape[1],)
        np.testing.assert_array_equal(first, second)
        assert max(int(np.sum(first == g)) for g in set(first.tolist())) <= 20

    def test_pathway_group_sparse_lasso_returns_selector_metadata(self):
        X, y = self._make_signed_proxy_data()

        results, all_scores = pathway_group_sparse_lasso_selection(
            X,
            y,
            n_target_features=4,
            n_groups=3,
            max_group_size=3,
            alpha=0.05,
            random_state=5,
        )

        assert results["pathway_group_sparse_lasso"] is True
        assert results["group_discovery_method"] == "signed_within_class_laplacian"
        assert results["selected_indices"].size == 4
        assert len(results["group_labels"]) == X.shape[1]
        assert len(all_scores) == X.shape[1]
        assert max(
            int(np.sum(np.asarray(results["group_labels"]) == g))
            for g in set(results["group_labels"])
        ) <= 3

    def test_pathway_group_sparse_lasso_runs_through_feature_selector_registry(self):
        X, y = self._make_signed_proxy_data()
        selector = FeatureSelector(
            enabled_methods={"pathway_group_sparse_lasso"},
            selection_strategy="legacy_voting",
            pathway_group_sparse_lasso_n_groups=3,
            pathway_group_sparse_lasso_max_group_size=3,
            group_sparse_lasso_alpha=0.05,
            random_state=5,
        )

        X_selected, result = selector.fit_transform(X, y, n_final_features=4, return_result_object=True)

        assert "pathway_group_sparse_lasso" in result.method_results
        method_result = result.method_results["pathway_group_sparse_lasso"]
        assert method_result["pathway_group_sparse_lasso"] is True
        assert X_selected.shape[0] == X.shape[0]
        assert 1 <= X_selected.shape[1] <= 4
        np.testing.assert_array_equal(selector.get_selected_features_indices(), result.selected_feature_indices)

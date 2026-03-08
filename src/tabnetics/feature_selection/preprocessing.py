"""Preprocessing utilities for feature selection."""
import numpy as np


def remove_correlated_features(X_df, threshold=0.90):
    """Remove highly correlated features. Returns (filtered_df, dropped_names)."""
    if X_df.shape[1] == 0:
        return X_df.copy(), []
    if X_df.shape[0] < 2:
        # Correlation is undefined for <2 rows; keep all columns.
        return X_df.copy(), []

    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_names = [column for column in upper.columns if any(upper[column] > threshold)]
    remaining_names = [col for col in X_df.columns if col not in to_drop_names]
    return X_df[remaining_names], to_drop_names

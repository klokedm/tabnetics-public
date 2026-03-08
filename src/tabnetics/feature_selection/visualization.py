"""Visualization utilities for feature selection."""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def build_feature_importance_figure(feature_votes, selected_indices):
    """Build a bar chart of feature importance by ensemble voting."""
    if plt is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    indices = np.arange(len(feature_votes))
    selected_set = set(int(i) for i in np.asarray(selected_indices, dtype=int).ravel().tolist())
    colors = ['green' if i in selected_set else 'gray' for i in indices]

    ax.bar(indices, feature_votes, color=colors)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Total Votes')
    ax.set_title('Feature Importance by Ensemble Voting')

    for idx in np.asarray(selected_indices, dtype=int).ravel()[:10]:
        idx_int = int(idx)
        if 0 <= idx_int < len(feature_votes):
            ax.text(idx_int, feature_votes[idx_int], f'{idx_int}', ha='center', va='bottom')

    fig.tight_layout()
    return fig


def close_feature_importance_figure(fig) -> None:
    """Close a feature-importance figure when matplotlib is available."""
    if plt is None or fig is None:
        return
    try:
        if int(fig.number) in plt.get_fignums():
            plt.close(fig)
    except Exception:
        pass

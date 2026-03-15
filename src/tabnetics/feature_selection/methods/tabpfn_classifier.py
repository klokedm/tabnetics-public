"""TabPFN benchmark classifier wrapper (VAL12_Suggestions §4.2).

Provides a size-guarded sklearn-compatible wrapper around TabPFN for
use as a post-FS benchmark reference.  TabPFN is a prior-fitted network
for small tabular classification — it requires **no training** and works
best with ≤100 features and ≤1,000 samples.

Usage::

    from tabnetics.feature_selection.methods.tabpfn_classifier import (
        TabPFNBenchmarkClassifier,
        TABPFN_AVAILABLE,
    )

    if TABPFN_AVAILABLE:
        clf = TabPFNBenchmarkClassifier(random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)

The wrapper silently falls back to a stratified-dummy classifier when
the input exceeds TabPFN's constraints or when the ``tabpfn`` package
is not installed.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted

# ------------------------------------------------------------------
# Optional dependency
# ------------------------------------------------------------------

try:
    from tabpfn import TabPFNClassifier as _TabPFNClassifier

    TABPFN_AVAILABLE = True
except Exception:  # pragma: no cover
    _TabPFNClassifier = None  # type: ignore[assignment,misc]
    TABPFN_AVAILABLE = False

# ------------------------------------------------------------------
# Size constraints
# ------------------------------------------------------------------

MAX_FEATURES: int = 100
MAX_SAMPLES: int = 1_000


# ------------------------------------------------------------------
# Wrapper
# ------------------------------------------------------------------


class TabPFNBenchmarkClassifier(BaseEstimator, ClassifierMixin):
    """Size-guarded TabPFN wrapper for benchmark-lane evaluation.

    Falls back to ``DummyClassifier(strategy='stratified')`` when:

    - ``tabpfn`` is not installed,
    - ``n_features > max_features``, or
    - ``n_samples > max_samples``.

    Parameters
    ----------
    max_features : int
        Maximum number of features TabPFN can handle (default 100).
    max_samples : int
        Maximum number of training samples (default 1000).
    random_state : int or None
        Random seed.
    device : str
        Torch device for TabPFN (default ``"auto"``).
    """

    def __init__(
        self,
        *,
        max_features: int = MAX_FEATURES,
        max_samples: int = MAX_SAMPLES,
        random_state: Optional[int] = None,
        device: str = "auto",
    ):
        self.max_features = int(max(1, max_features))
        self.max_samples = int(max(1, max_samples))
        self.random_state = random_state
        # Force CPU when no NVIDIA GPU driver is detected to prevent
        # cuBLAS deadlock in multi-process workloads.
        import os as _os_tabpfn
        if device == "auto" and not (
            _os_tabpfn.path.isdir("/proc/driver/nvidia")
            or any(_os_tabpfn.path.exists(f"/dev/nvidia{i}") for i in range(4))
        ):
            device = "cpu"
        self.device = device

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNBenchmarkClassifier":
        """Fit the classifier (or fall back to dummy)."""
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_features_in_ = n_features

        # Decide whether to use real TabPFN or fall back.
        self.exceeded_limits_ = False
        self.fallback_reason_: Optional[str] = None
        self._inner: BaseEstimator

        if not TABPFN_AVAILABLE:
            self.fallback_reason_ = "tabpfn package not installed"
            self.exceeded_limits_ = True
        elif n_features > self.max_features:
            self.fallback_reason_ = (
                f"n_features={n_features} exceeds max_features={self.max_features}"
            )
            self.exceeded_limits_ = True
        elif n_samples > self.max_samples:
            self.fallback_reason_ = (
                f"n_samples={n_samples} exceeds max_samples={self.max_samples}"
            )
            self.exceeded_limits_ = True

        if self.exceeded_limits_:
            warnings.warn(
                f"TabPFN fallback to DummyClassifier: {self.fallback_reason_}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._inner = DummyClassifier(
                strategy="stratified",
                random_state=self.random_state,
            )
            self._inner.fit(X, y)
            return self

        # Try to instantiate real TabPFN with compatible kwargs.
        inner = self._try_build_tabpfn()
        if inner is None:
            self.fallback_reason_ = "TabPFN instantiation failed"
            self.exceeded_limits_ = True
            self._inner = DummyClassifier(
                strategy="stratified",
                random_state=self.random_state,
            )
            self._inner.fit(X, y)
            return self

        inner.fit(X, y)
        self._inner = inner
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        check_is_fitted(self, "_inner")
        return self._inner.predict(np.asarray(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self, "_inner")
        return self._inner.predict_proba(np.asarray(X))

    @property
    def is_using_tabpfn(self) -> bool:
        """Whether the real TabPFN backend is active (vs fallback)."""
        return not getattr(self, "exceeded_limits_", True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _try_build_tabpfn(self):
        """Try several kwarg combos for TabPFN compatibility."""
        if _TabPFNClassifier is None:
            return None  # pragma: no cover
        attempts = [
            {"random_state": self.random_state, "device": self.device},
            {"device": self.device},
            {"random_state": self.random_state},
            {},
        ]
        for kwargs in attempts:
            try:
                return _TabPFNClassifier(**kwargs)
            except TypeError:
                continue
            except Exception:
                continue
        return None

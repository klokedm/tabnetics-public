import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabnetics.classification.backends import (
    ClassifierOracle,
    MNPOClassifierBackend,
    PLSDAClassifier,
    SklearnBackend,
    _HardVotingPredictAdapter,
    _LabelEncodedEstimator,
    _SoftVotingPredictAdapter,
    REGIME_HDLSS_EXTREME,
    REGIME_HDLSS_MODERATE,
    REGIME_STANDARD,
    classify_regime,
)


def _toy_data(n_samples=90, n_features=24, n_classes=3, seed=0):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=max(6, n_classes * 2),
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=seed,
    )
    classes, counts = np.unique(y, return_counts=True)
    return np.asarray(X, dtype=float), np.asarray(y), int(classes.size), np.asarray(counts, dtype=int)


class _ColumnVectorPredictClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self._lr = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
        self._lr.fit(X, y)
        self.classes_ = np.asarray(self._lr.classes_)
        return self

    def predict(self, X):
        return np.asarray(self._lr.predict(X)).reshape(-1, 1)

    def predict_proba(self, X):
        return self._lr.predict_proba(X)


class _ScoreMatrixPredictClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self._lr = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
        self._lr.fit(X, y)
        self.classes_ = np.asarray(self._lr.classes_)
        return self

    def predict(self, X):
        return np.asarray(self._lr.predict_proba(X), dtype=float)

    def predict_proba(self, X):
        return self._lr.predict_proba(X)


def test_classify_regime_thresholds():
    assert classify_regime(40, 20) == REGIME_HDLSS_EXTREME
    assert classify_regime(100, 6000) == REGIME_HDLSS_MODERATE  # p/n = 60
    assert classify_regime(260, 5000) == REGIME_STANDARD
    assert classify_regime(80, 60000) == REGIME_HDLSS_EXTREME  # p/n = 750


def test_classifier_oracle_run_smoke():
    X, y, _, _ = _toy_data(n_samples=96, n_features=24, n_classes=3, seed=7)
    backend = SklearnBackend(candidate_names=("lr", "svm_rbf", "nb", "dlda", "nsc"))
    candidates = backend._build_candidates(X_train=X, y_train=y, seed=13)
    candidate_names = ("lr", "svm_rbf", "nb", "dlda", "nsc")

    oracle = ClassifierOracle(
        include_calibration=True,
        include_james_stein=True,
        enable_hoeffding_racing=True,
        hoeffding_delta=0.10,
        enable_bbc=True,
        bbc_bootstrap_rounds=32,
        bbc_ci_level=0.90,
    )
    out = oracle.run(
        candidates=candidates,
        candidate_names=candidate_names,
        X=X,
        y=y,
        seed=19,
        cv_splits=5,
        top_k=2,
    )

    assert 1 <= len(out.get("selected_names") or []) <= 2
    assert "performance" in dict(out.get("oracle_weights") or {})
    assert isinstance(out.get("candidate_stats"), dict)
    assert int((out.get("race_meta") or {}).get("n_splits", 0)) >= 2


def test_mnpo_backend_regime_gates_tree_candidates():
    X, y, n_classes, counts = _toy_data(n_samples=42, n_features=2600, n_classes=2, seed=11)
    backend = MNPOClassifierBackend(
        candidate_names=("lr", "rf", "lgbm", "extra_tree", "catboost", "nsc", "svm_linear"),
        oracle_k=1,
        use_per_family_flaml=False,
    )
    _, model_name, _, _, _, meta = backend.fit_and_select(
        X,
        y,
        seed=23,
        n_classes=n_classes,
        class_counts=counts,
    )
    assert str(meta.get("classification_regime")) == REGIME_HDLSS_EXTREME
    dropped = set(str(x) for x in (meta.get("classification_regime_dropped_candidates") or []))
    assert {"rf", "lgbm", "extra_tree", "catboost"} & dropped
    assert str(model_name).startswith("mnpo_")


def test_mnpo_backend_non_native_family_falls_back_without_flaml():
    X, y, n_classes, counts = _toy_data(n_samples=88, n_features=20, n_classes=3, seed=21)
    backend = MNPOClassifierBackend(
        candidate_names=("nsc",),
        oracle_k=1,
        use_per_family_flaml=True,
        flaml_time_budget=60,
    )
    _, model_name, _, _, _, meta = backend.fit_and_select(
        X,
        y,
        seed=31,
        n_classes=n_classes,
        class_counts=counts,
    )
    assert str(model_name) == "nsc"
    assert str(meta.get("mnpo_hpo_mode")) == "per_family_single"
    hpo_meta = dict(meta.get("mnpo_hpo_meta") or {})
    assert hpo_meta.get("mnpo_flaml_fallback_reason") == "family_not_supported_by_flaml"


def test_mnpo_backend_optional_ensemble_mode_builds_voting_classifier():
    X, y, n_classes, counts = _toy_data(n_samples=92, n_features=26, n_classes=3, seed=33)
    backend = MNPOClassifierBackend(
        candidate_names=("nsc", "pls_da_classifier"),
        oracle_k=2,
        enable_ensemble=True,
        use_per_family_flaml=True,
        flaml_time_budget=60,
    )
    model, model_name, _, _, _, meta = backend.fit_and_select(
        X,
        y,
        seed=41,
        n_classes=n_classes,
        class_counts=counts,
    )
    assert isinstance(model, VotingClassifier)
    assert str(model_name).startswith("mnpo_ensemble_")
    assert str(meta.get("mnpo_hpo_mode")) == "per_family_ensemble"


def test_custom_wrappers_report_classifier_tags():
    wrapped = _LabelEncodedEstimator(
        LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
    )
    pls_pipeline = make_pipeline(StandardScaler(), PLSDAClassifier(n_components=2, scale=True))
    assert is_classifier(wrapped) is True
    assert is_classifier(pls_pipeline) is True


def test_voting_classifier_accepts_wrapped_and_pls_pipeline_estimators():
    X, y, _, _ = _toy_data(n_samples=84, n_features=18, n_classes=3, seed=101)
    model = VotingClassifier(
        estimators=[
            (
                "wrapped_lr",
                _LabelEncodedEstimator(
                    LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
                ),
            ),
            (
                "pls",
                make_pipeline(StandardScaler(), PLSDAClassifier(n_components=3, scale=True)),
            ),
        ],
        voting="hard",
        n_jobs=1,
    )
    model.fit(X, y)
    y_pred = np.asarray(model.predict(X)).ravel()
    assert y_pred.shape[0] == X.shape[0]


def test_hard_voting_predict_adapter_handles_column_and_score_matrix_predict_outputs():
    X, y, _, _ = _toy_data(n_samples=84, n_features=18, n_classes=3, seed=202)
    model = VotingClassifier(
        estimators=[
            ("column", _HardVotingPredictAdapter(_ColumnVectorPredictClassifier())),
            ("score_matrix", _HardVotingPredictAdapter(_ScoreMatrixPredictClassifier())),
            (
                "wrapped_lr",
                _LabelEncodedEstimator(
                    LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
                ),
            ),
        ],
        voting="hard",
        n_jobs=1,
    )
    model.fit(X, y)
    y_pred = np.asarray(model.predict(X)).ravel()
    assert y_pred.shape[0] == X.shape[0]
    assert set(np.unique(y_pred)).issubset(set(np.unique(y)))


def test_label_encoded_estimator_coerces_score_matrix_predict_output():
    X, y, _, _ = _toy_data(n_samples=72, n_features=16, n_classes=3, seed=303)
    model = _LabelEncodedEstimator(_ScoreMatrixPredictClassifier())
    model.fit(X, y)
    y_pred = np.asarray(model.predict(X)).ravel()
    assert y_pred.shape[0] == X.shape[0]
    assert set(np.unique(y_pred)).issubset(set(np.unique(y)))


# --- B2: Soft Voting Predict Adapter ---


class _NoProbClassifier(ClassifierMixin, BaseEstimator):
    """Classifier that only has predict(), no predict_proba()."""

    def fit(self, X, y):
        self._lr = LogisticRegression(max_iter=2000, solver="lbfgs")
        self._lr.fit(X, y)
        self.classes_ = np.asarray(self._lr.classes_)
        return self

    def predict(self, X):
        return self._lr.predict(X)


def test_soft_voting_predict_adapter_delegates_predict_proba():
    X, y, _, _ = _toy_data(n_samples=80, n_features=16, n_classes=3, seed=401)
    base = LogisticRegression(max_iter=2000, solver="lbfgs")
    adapter = _SoftVotingPredictAdapter(base)
    adapter.fit(X, y)
    proba = adapter.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_soft_voting_predict_adapter_fallback_for_no_proba_estimator():
    X, y, _, _ = _toy_data(n_samples=80, n_features=16, n_classes=3, seed=402)
    base = _NoProbClassifier()
    adapter = _SoftVotingPredictAdapter(base)
    adapter.fit(X, y)
    proba = adapter.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    # One-hot fallback: each row should have exactly a single 1.0.
    assert np.all(proba.max(axis=1) == 1.0)
    assert np.all(proba.sum(axis=1) == 1.0)


# --- B8: Incumbent early stopping in ClassifierOracle ---


def test_classifier_oracle_incumbent_early_stopping():
    X, y, _, _ = _toy_data(n_samples=96, n_features=24, n_classes=3, seed=501)
    backend = SklearnBackend(candidate_names=("lr", "svm_rbf", "nb", "dlda", "nsc"))
    candidates = backend._build_candidates(X_train=X, y_train=y, seed=13)
    candidate_names = ("lr", "svm_rbf", "nb", "dlda", "nsc")

    oracle = ClassifierOracle(
        include_calibration=True,
        include_james_stein=True,
        enable_hoeffding_racing=True,
        hoeffding_delta=0.10,
        enable_bbc=True,
        bbc_bootstrap_rounds=32,
        bbc_ci_level=0.90,
        incumbent_early_stopping=True,
    )
    out = oracle.run(
        candidates=candidates,
        candidate_names=candidate_names,
        X=X,
        y=y,
        seed=19,
        cv_splits=5,
        top_k=2,
    )
    assert 1 <= len(out.get("selected_names") or []) <= 2
    race_meta = out.get("race_meta") or {}
    assert int(race_meta.get("n_splits", 0)) >= 2


# --- B3: Candidate pruning by marginal contribution ---


def test_classifier_oracle_candidate_pruning():
    X, y, _, _ = _toy_data(n_samples=96, n_features=24, n_classes=3, seed=601)
    backend = SklearnBackend(candidate_names=("lr", "svm_rbf", "nb", "dlda", "nsc"))
    candidates = backend._build_candidates(X_train=X, y_train=y, seed=13)
    candidate_names = ("lr", "svm_rbf", "nb", "dlda", "nsc")

    oracle = ClassifierOracle(
        include_calibration=True,
        include_james_stein=True,
        enable_hoeffding_racing=True,
        hoeffding_delta=0.10,
        enable_bbc=True,
        bbc_bootstrap_rounds=32,
        bbc_ci_level=0.90,
        candidate_pruning=True,
        candidate_pruning_threshold=0.0,
    )
    out = oracle.run(
        candidates=candidates,
        candidate_names=candidate_names,
        X=X,
        y=y,
        seed=19,
        cv_splits=5,
        top_k=2,
    )
    assert 1 <= len(out.get("selected_names") or []) <= 2
    race_meta = out.get("race_meta") or {}
    # Pruning metadata should be present.
    assert "candidate_pruning_applied" in race_meta


# --- B1: Greedy ensemble selection via MNPOClassifierBackend ---


def test_mnpo_backend_greedy_ensemble_selection():
    X, y, n_classes, counts = _toy_data(n_samples=92, n_features=26, n_classes=3, seed=701)
    backend = MNPOClassifierBackend(
        candidate_names=("nsc", "pls_da_classifier", "lr"),
        oracle_k=2,
        enable_ensemble=True,
        greedy_ensemble=True,
        greedy_ensemble_rounds=5,
        use_per_family_flaml=True,
        flaml_time_budget=60,
    )
    model, model_name, _, _, _, meta = backend.fit_and_select(
        X,
        y,
        seed=41,
        n_classes=n_classes,
        class_counts=counts,
    )
    assert isinstance(model, VotingClassifier)
    assert str(model_name).startswith("mnpo_ensemble_")


# --- B2: Soft voting ensemble via MNPOClassifierBackend ---


def test_mnpo_backend_soft_voting_ensemble():
    X, y, n_classes, counts = _toy_data(n_samples=92, n_features=26, n_classes=3, seed=801)
    backend = MNPOClassifierBackend(
        candidate_names=("lr", "nb"),
        oracle_k=2,
        enable_ensemble=True,
        ensemble_voting_mode="soft",
        use_per_family_flaml=True,
        flaml_time_budget=60,
    )
    model, model_name, _, _, _, meta = backend.fit_and_select(
        X,
        y,
        seed=51,
        n_classes=n_classes,
        class_counts=counts,
    )
    assert isinstance(model, VotingClassifier)
    assert str(model.voting) == "soft"
    assert str(model_name).startswith("mnpo_ensemble_")

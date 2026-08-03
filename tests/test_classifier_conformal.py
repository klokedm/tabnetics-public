import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tabnetics.feature_selection.conformal import compute_split_conformal_sets


def _toy_binary(seed=0, n_samples=120, n_features=24):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=2,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return np.asarray(X, dtype=float), np.asarray(y)


def test_split_conformal_sets_smoke_binary():
    X, y = _toy_binary(seed=3, n_samples=120, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=11, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced"),
    )
    model.fit(X_train, y_train)

    out = compute_split_conformal_sets(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        y_eval=y_test,
        alpha=0.10,
        calibration_fraction=0.25,
        min_calibration=20,
        seed=19,
        include_prediction_sets=False,
    )

    assert out["classifier_conformal_enabled"] is True
    assert out["classifier_conformal_applied"] is True
    assert out["classifier_conformal_skip_reason"] == ""
    assert 0.0 <= float(out["classifier_conformal_coverage"]) <= 1.0
    assert float(out["classifier_conformal_set_size_mean"]) >= 1.0
    assert 0.0 <= float(out["classifier_conformal_singleton_rate"]) <= 1.0
    assert int(out["classifier_conformal_calibration_size"]) >= 20
    assert out["classifier_conformal_score_source"] == "predict_proba"
    assert out["classifier_conformal_conformity_kind"] == "probability_matrix"
    assert out["classifier_conformal_source_consistent"] is True


def test_split_conformal_sets_skips_when_train_too_small():
    X, y = _toy_binary(seed=5, n_samples=32, n_features=12)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)

    out = compute_split_conformal_sets(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        y_eval=None,
        alpha=0.10,
        calibration_fraction=0.5,
        min_calibration=30,
        seed=17,
        include_prediction_sets=False,
    )

    assert out["classifier_conformal_applied"] is False
    assert out["classifier_conformal_skip_reason"] in {
        "insufficient_train_samples",
        "insufficient_calibration_size",
    }


def test_split_conformal_sets_can_emit_prediction_sets():
    X, y = _toy_binary(seed=13, n_samples=100, n_features=16)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=29, stratify=y
    )
    model = LogisticRegression(max_iter=4000, solver="lbfgs")
    model.fit(X_train, y_train)

    out = compute_split_conformal_sets(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        y_eval=None,
        alpha=0.15,
        calibration_fraction=0.30,
        min_calibration=15,
        seed=23,
        include_prediction_sets=True,
    )

    assert out["classifier_conformal_enabled"] is True
    if out["classifier_conformal_applied"]:
        sets = list(out["classifier_conformal_prediction_sets"])
        assert len(sets) == X_test.shape[0]
        assert all(len(s) >= 1 for s in sets)


def test_split_conformal_sets_uses_probability_enabled_svc_scores():
    X, y = _toy_binary(seed=17, n_samples=140, n_features=18)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=31, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=31),
    )

    out = compute_split_conformal_sets(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        y_eval=y_test,
        alpha=0.10,
        calibration_fraction=0.25,
        min_calibration=15,
        seed=31,
        include_prediction_sets=True,
        include_score_source=True,
    )

    assert out["classifier_conformal_applied"] is True
    assert out["classifier_conformal_score_source"] == "predict_proba"
    assert out["classifier_conformal_used_predict_proba"] is True
    assert out["classifier_conformal_model_supports_predict_proba"] is True
    assert len(out["classifier_conformal_prediction_sets"]) == X_test.shape[0]


def test_split_conformal_sets_falls_back_for_svc_without_probability():
    X, y = _toy_binary(seed=19, n_samples=140, n_features=18)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=37, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=1.0, probability=False, random_state=37),
    )

    out = compute_split_conformal_sets(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        y_eval=y_test,
        alpha=0.10,
        calibration_fraction=0.25,
        min_calibration=15,
        seed=37,
        include_prediction_sets=False,
        include_score_source=True,
    )

    assert out["classifier_conformal_applied"] is True
    assert out["classifier_conformal_score_source"] == "decision_function_sigmoid"
    assert out["classifier_conformal_used_predict_proba"] is False


def test_split_conformal_sets_uses_soft_vote_predict_proba():
    X, y = _toy_binary(seed=23, n_samples=140, n_features=18)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=41, stratify=y
    )
    model = VotingClassifier(
        estimators=[
            ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
            (
                "svc",
                make_pipeline(
                    StandardScaler(),
                    SVC(kernel="linear", probability=True, random_state=41),
                ),
            ),
            ("nb", GaussianNB()),
        ],
        voting="soft",
    )

    out = compute_split_conformal_sets(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        y_eval=y_test,
        alpha=0.10,
        calibration_fraction=0.25,
        min_calibration=15,
        seed=41,
        include_prediction_sets=False,
        include_score_source=True,
    )

    assert out["classifier_conformal_applied"] is True
    assert out["classifier_conformal_score_source"] == "predict_proba"
    assert out["classifier_conformal_used_predict_proba"] is True

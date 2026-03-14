"""Cross-validation strategy helpers for feature selection."""
import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional

from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    RepeatedStratifiedKFold,
    RepeatedKFold,
)
from sklearn.linear_model import Ridge

from tabnetics.core.runtime import get_sklearn_n_jobs as _get_sklearn_n_jobs
try:
    from tabnetics.core.compat import make_logistic_regression
except Exception as exc:
    from tabnetics.core.compat import make_logistic_regression  # type: ignore
from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class FoldLeakageError(RuntimeError):
    """Raised when evaluation-time settings violate fold-hygiene constraints."""


@dataclass(frozen=True)
class CVEvaluationContext:
    """Explicit evaluation context contract for fold-hygiene controls.

    Parameters
    ----------
    purpose:
        Intended context for scoring path, e.g. ``"evaluation_fold"``.
    allow_learned_model_aggregation:
        Whether learned/adaptive model aggregation is allowed in this context.
        Must remain ``False`` for evaluation folds.
    """

    purpose: str = "evaluation_fold"
    allow_learned_model_aggregation: bool = False


def get_cv_strategy(n_samples, y, problem_type, random_state, inner_cv_splits, inner_cv_repeats, purpose='method_internal'):
    """
    Unified CV strategy selection (P1-3: replaces inconsistent LOOCV vs RepeatedStratifiedKFold).

    Purpose parameter allows different strategies for different contexts:
    - 'method_internal': For individual method CV (boruta, RFE, etc.)
    - 'mnpo_evaluation': For MNPO candidate evaluation
    - 'wrapper': For wrapper methods

    Default strategy:
    - n < 20: LOOCV (too few samples for k-fold)
    - n >= 20: RepeatedStratifiedKFold(n_splits=5, n_repeats=3) if possible,
               fallback to RepeatedKFold if stratification impossible

    Rationale: LOOCV on n=100+ is wasteful and high-variance. K-fold is more stable.
    """
    if n_samples < 20:
        return LeaveOneOut()

    if problem_type == 'classification':
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            n_splits = min(3, n_samples)
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        min_count = int(np.min(counts))
        if min_count < 2:
            # Class-sparse fallback: stratified CV invalid
            n_splits_val = int(max(2, min(inner_cv_splits, n_samples - 1)))
            return RepeatedKFold(
                n_splits=n_splits_val,
                n_repeats=max(1, inner_cv_repeats),
                random_state=random_state,
            )

        # Standard stratified k-fold
        max_splits = int(min_count)
        n_splits_val = int(max(2, min(inner_cv_splits, max_splits)))
        return RepeatedStratifiedKFold(
            n_splits=n_splits_val,
            n_repeats=max(1, inner_cv_repeats),
            random_state=random_state,
        )

    # Regression: use repeated k-fold
    n_splits_val = int(max(2, min(inner_cv_splits, n_samples - 1)))
    return RepeatedKFold(
        n_splits=n_splits_val,
        n_repeats=max(1, inner_cv_repeats),
        random_state=random_state,
    )


def get_inner_cv_splits(X, y, problem_type, random_state, inner_cv_splits, inner_cv_repeats):
    """Create repeated inner CV splits for noisy pairwise preference estimation."""
    n_samples = X.shape[0]
    if n_samples < 2:
        return []
    if n_samples < 4:
        kf = KFold(n_splits=2, shuffle=True, random_state=random_state)
        return list(kf.split(X))

    if problem_type == 'classification':
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            kf = KFold(n_splits=min(3, n_samples), shuffle=True, random_state=random_state)
            return list(kf.split(X))

        min_count = int(np.min(counts))
        if min_count < 2:
            # Class-sparse fallback: stratified CV is invalid when a class has <2 samples.
            n_splits_val = int(max(2, min(inner_cv_splits, n_samples - 1)))
            splitter = RepeatedKFold(
                n_splits=n_splits_val,
                n_repeats=max(1, inner_cv_repeats),
                random_state=random_state,
            )
            return list(splitter.split(X))

        max_splits = int(min_count)
        n_splits_val = int(max(2, min(inner_cv_splits, max_splits)))
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits_val,
            n_repeats=max(1, inner_cv_repeats),
            random_state=random_state,
        )
        return list(splitter.split(X, y))

    n_splits_val = int(max(2, min(inner_cv_splits, n_samples - 1)))
    splitter = RepeatedKFold(
        n_splits=n_splits_val,
        n_repeats=max(1, inner_cv_repeats),
        random_state=random_state,
    )
    return list(splitter.split(X))


def safe_balanced_accuracy(y_true, y_pred):
    """Safe balanced accuracy that handles edge cases."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = np.unique(y_true_arr)
    if labels.size == 0:
        return 0.0
    if labels.size == 1:
        return float(np.mean(y_pred_arr == labels[0]))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return float(recall_score(y_true_arr, y_pred_arr, labels=labels, average='macro', zero_division=0))


def safe_macro_f1(y_true, y_pred):
    """Safe macro F1 that handles edge cases."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = np.unique(y_true_arr)
    if labels.size == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return float(f1_score(y_true_arr, y_pred_arr, labels=labels, average='macro', zero_division=0))


def augment_training_data(X_train, y_train, problem_type, random_state):
    """
    Lightweight counterfactual-style augmentation:
    class-balancing with small Gaussian perturbations for minority classes.
    """
    if problem_type != 'classification' or X_train.shape[0] < 6:
        return X_train, y_train

    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2:
        return X_train, y_train

    min_count = int(np.min(counts))
    max_count = int(np.max(counts))
    if min_count <= 0 or (max_count / max(min_count, 1)) < 1.2:
        return X_train, y_train

    rng = np.random.default_rng(random_state)
    feature_scale = np.std(X_train, axis=0, ddof=1)
    feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > 1e-8), feature_scale, 1.0)

    synthetic_X = []
    synthetic_y = []
    for cls, cls_count in zip(classes, counts):
        deficit = max_count - int(cls_count)
        if deficit <= 0:
            continue
        cls_idx = np.where(y_train == cls)[0]
        sampled_idx = rng.choice(cls_idx, size=deficit, replace=True)
        noise = rng.normal(0.0, 0.02, size=(deficit, X_train.shape[1])) * feature_scale
        synthetic_X.append(X_train[sampled_idx] + noise)
        synthetic_y.append(np.full(deficit, cls))

    if not synthetic_X:
        return X_train, y_train

    X_aug = np.vstack([X_train] + synthetic_X)
    y_aug = np.concatenate([y_train] + synthetic_y)
    return X_aug, y_aug


def resolve_performance_weights(y_train, problem_type, performance_balanced_weight,
                                performance_macro_f1_weight, performance_use_adaptive_imbalance,
                                performance_imbalance_ratio_trigger, performance_min_classes_for_adaptive):
    """
    Resolve performance-oracle weights; optionally increase macro-F1 emphasis
    on highly imbalanced multiclass folds.
    """
    w_bal = float(max(0.0, performance_balanced_weight))
    w_f1 = float(max(0.0, performance_macro_f1_weight))
    if (w_bal + w_f1) <= 1e-12:
        return 1.0, 0.0

    if (
        problem_type != 'classification'
        or not performance_use_adaptive_imbalance
    ):
        return w_bal, w_f1

    classes, counts = np.unique(y_train, return_counts=True)
    if classes.size < performance_min_classes_for_adaptive:
        return w_bal, w_f1
    min_count = int(np.min(counts))
    if min_count <= 0:
        return w_bal, w_f1
    imbalance_ratio = float(np.max(counts) / max(min_count, 1))
    if imbalance_ratio < performance_imbalance_ratio_trigger:
        return w_bal, w_f1

    ratio_span = max(0.25, performance_imbalance_ratio_trigger)
    strength = min(1.0, (imbalance_ratio - performance_imbalance_ratio_trigger) / ratio_span)
    shift = 0.25 * strength
    w_f1_adj = w_f1 + shift
    w_bal_adj = max(0.05, w_bal - shift)
    denom = w_bal_adj + w_f1_adj
    if denom <= 1e-12:
        return w_bal, w_f1
    return float(w_bal_adj), float(w_f1_adj)


def fit_and_score_fold(X_train, y_train, X_val, y_val, problem_type, random_state,
                       performance_balanced_weight, performance_macro_f1_weight,
                       performance_use_adaptive_imbalance, performance_imbalance_ratio_trigger,
                       performance_min_classes_for_adaptive, model_cv_lr_max_iter=2000):
    """Fit a low-capacity downstream model and return scalar score + prediction signal."""
    if problem_type == 'classification':
        unique_train = np.unique(y_train)
        if unique_train.size < 2:
            preds = np.full(y_val.shape[0], unique_train[0] if unique_train.size == 1 else 0)
            score = safe_balanced_accuracy(y_val, preds)
            return float(score), preds.astype(float)

        model = make_logistic_regression(
            solver='lbfgs',
            penalty='l2',
            C=1.0,
            max_iter=model_cv_lr_max_iter,
            class_weight='balanced',
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        bal_acc = safe_balanced_accuracy(y_val, preds)
        macro_f1 = safe_macro_f1(y_val, preds)
        w_bal, w_f1 = resolve_performance_weights(
            y_train, problem_type, performance_balanced_weight, performance_macro_f1_weight,
            performance_use_adaptive_imbalance, performance_imbalance_ratio_trigger,
            performance_min_classes_for_adaptive,
        )
        denom = max(1e-12, w_bal + w_f1)
        score = float((w_bal * bal_acc + w_f1 * macro_f1) / denom)

        signal = preds.astype(float)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)
            if probs.ndim == 2 and probs.shape[1] == 2 and np.unique(y_val).size > 1:
                try:
                    auc = roc_auc_score(y_val, probs[:, 1])
                    score = 0.8 * score + 0.2 * float(auc)
                except Exception as exc:
                    pass
                signal = probs[:, 1]
            elif probs.ndim == 2 and probs.shape[1] > 2:
                # Use predicted labels as a compact multiclass signal; this
                # is more informative for diversity/PID diagnostics than max-prob.
                signal = preds.astype(float)

        return float(score), np.asarray(signal, dtype=float).ravel()

    model = Ridge(alpha=1.0, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    try:
        r2 = r2_score(y_val, preds)
    except Exception as exc:
        r2 = 0.0
    denom = np.std(y_train) + 1e-8
    score = -(0.6 * rmse / denom + 0.4 * mae / denom) + 0.1 * r2
    return float(score), np.asarray(preds, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Multi-classifier evaluation proxy (T-001)
# ---------------------------------------------------------------------------

def _get_eval_model(model_key: str, random_state: int, max_iter: int = 2000):
    """Return a classifier instance for the given model key.

    Only fixed-weight aggregation is used (no learned weights).
    See ArchitectureRefactor.md §14, RISK-1.
    """
    if model_key == "lr_l2":
        return make_logistic_regression(
            solver='lbfgs', penalty='l2', C=1.0,
            max_iter=max_iter, class_weight='balanced',
            random_state=random_state,
        )
    elif model_key == "linear_svc":
        from sklearn.svm import LinearSVC
        return LinearSVC(
            C=1.0, class_weight='balanced',
            max_iter=max_iter, random_state=random_state,
            dual='auto',
        )
    elif model_key == "rf_small":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=50, max_depth=5,
            class_weight='balanced',
            random_state=random_state, n_jobs=_get_sklearn_n_jobs(),
        )
    else:
        raise ValueError(f"Unknown eval model: {model_key}")


def _score_fitted_model(model, X_val, y_val, w_bal, w_f1):
    """Score a single already-fitted classification model.

    Replicates the same scoring logic as ``fit_and_score_fold`` for
    classification: weighted balanced-accuracy / macro-F1, with an
    optional AUC bonus for binary tasks when ``predict_proba`` is
    available.

    Returns ``(score, signal)``.
    """
    preds = model.predict(X_val)
    bal_acc = safe_balanced_accuracy(y_val, preds)
    macro_f1_val = safe_macro_f1(y_val, preds)
    denom = max(1e-12, w_bal + w_f1)
    score = float((w_bal * bal_acc + w_f1 * macro_f1_val) / denom)

    signal = preds.astype(float)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)
        if probs.ndim == 2 and probs.shape[1] == 2 and np.unique(y_val).size > 1:
            try:
                auc = roc_auc_score(y_val, probs[:, 1])
                score = 0.8 * score + 0.2 * float(auc)
            except Exception as exc:
                pass
            signal = probs[:, 1]
        elif probs.ndim == 2 and probs.shape[1] > 2:
            signal = preds.astype(float)

    return float(score), np.asarray(signal, dtype=float).ravel()


def fit_and_score_fold_multimodel(
    X_train, y_train, X_val, y_val,
    problem_type, random_state,
    eval_models, eval_aggregate, eval_cvar_alpha,
    performance_balanced_weight, performance_macro_f1_weight,
    performance_use_adaptive_imbalance,
    performance_imbalance_ratio_trigger,
    performance_min_classes_for_adaptive,
    model_cv_lr_max_iter=2000,
    eval_failure_strict_mode=False,
    eval_model_weight_strategy="fixed",
    evaluation_context: Optional[CVEvaluationContext] = None,
):
    """Multi-classifier fold scoring with fixed-weight aggregation.

    Each classifier is fitted independently on the training fold and
    scored independently on the validation fold.  Scores are combined
    with a fixed-weight aggregation (``mean``, ``min``, or ``cvar``).

    **No learned/adaptive weights** — Stage 1 only
    (see ArchitectureRefactor.md §14, RISK-1).

    Returns
    -------
    agg_score : float
        Aggregated score across classifiers.
    signal : np.ndarray
        Prediction signal from the LR model (backward compat) or the
        first model if LR is not in the model set.
    per_model_scores : dict[str, float]
        Per-model scores for diagnostics.
    """
    context = evaluation_context or CVEvaluationContext()
    model_weight_strategy = str(eval_model_weight_strategy or "fixed").strip().lower()
    if (
        context.purpose == "evaluation_fold"
        and (model_weight_strategy != "fixed")
        and (not context.allow_learned_model_aggregation)
    ):
        raise FoldLeakageError(
            "Learned/adaptive model aggregation is disallowed in evaluation folds; "
            "use eval_model_weight_strategy='fixed'."
        )

    # Regression: fall back to single-model scorer (Ridge)
    if problem_type != 'classification':
        score, signal = fit_and_score_fold(
            X_train, y_train, X_val, y_val, problem_type, random_state,
            performance_balanced_weight, performance_macro_f1_weight,
            performance_use_adaptive_imbalance,
            performance_imbalance_ratio_trigger,
            performance_min_classes_for_adaptive,
            model_cv_lr_max_iter,
        )
        return score, signal, {"_fallback_regression": float(score)}

    # Single-class edge case: replicate fit_and_score_fold behavior
    unique_train = np.unique(y_train)
    if unique_train.size < 2:
        preds = np.full(y_val.shape[0], unique_train[0] if unique_train.size == 1 else 0)
        score = safe_balanced_accuracy(y_val, preds)
        return float(score), preds.astype(float), {"_single_class": float(score)}

    # Resolve performance weights once (shared across all models)
    w_bal, w_f1 = resolve_performance_weights(
        y_train, problem_type,
        performance_balanced_weight, performance_macro_f1_weight,
        performance_use_adaptive_imbalance,
        performance_imbalance_ratio_trigger,
        performance_min_classes_for_adaptive,
    )

    eval_model_keys = tuple(str(mk) for mk in eval_models)
    per_model_scores = {}
    per_model_signals = {}
    failure_count_total = 0
    failure_count_by_model = {str(mk): 0 for mk in eval_model_keys}

    for model_key in eval_model_keys:
        model = _get_eval_model(model_key, random_state, model_cv_lr_max_iter)
        try:
            model.fit(X_train, y_train)
            m_score, m_signal = _score_fitted_model(model, X_val, y_val, w_bal, w_f1)
        except Exception as exc:
            if bool(eval_failure_strict_mode):
                raise
            m_score = 0.0
            m_signal = np.zeros(y_val.shape[0], dtype=float)
            failure_count_total += 1
            failure_count_by_model[str(model_key)] = int(failure_count_by_model.get(str(model_key), 0)) + 1
        per_model_scores[model_key] = float(m_score)
        per_model_signals[model_key] = m_signal

    per_model_scores["_evaluation_failures_total"] = float(failure_count_total)
    for mk, count in failure_count_by_model.items():
        per_model_scores[f"_evaluation_failures_{mk}"] = float(count)

    # Fixed-weight aggregation (no learned weights — RISK-1 mitigation)
    # Aggregate only actual model-score entries (exclude diagnostics counters).
    scores_arr = np.array([per_model_scores[mk] for mk in eval_model_keys], dtype=float)
    if eval_aggregate == "min":
        agg_score = float(np.min(scores_arr))
    elif eval_aggregate == "cvar":
        k = max(1, int(np.ceil(eval_cvar_alpha * len(scores_arr))))
        sorted_scores = np.sort(scores_arr)
        agg_score = float(np.mean(sorted_scores[:k]))
    else:  # default: "mean"
        agg_score = float(np.mean(scores_arr))

    # Use LR signal as primary (backward compat); fall back to first model
    if "lr_l2" in per_model_signals:
        signal = per_model_signals["lr_l2"]
    else:
        signal = per_model_signals[eval_models[0]]

    return float(agg_score), np.asarray(signal, dtype=float).ravel(), per_model_scores


def compute_feature_importance_uq(
    X,
    y,
    *,
    problem_type,
    random_state,
    inner_cv_splits,
    inner_cv_repeats,
    min_cv_folds=3,
    max_folds=8,
    model_cv_lr_max_iter=2000,
):
    """Estimate per-feature importance variance across CV folds.

    Returns a dictionary with mean/variance vectors and unstable-feature indices.
    This is reporting-only and does not affect feature selection decisions.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    n_features = int(X_arr.shape[1]) if X_arr.ndim == 2 else 0

    empty = {
        "importance_uq_enabled": True,
        "importance_uq_computed": False,
        "importance_uq_reason": "uninitialized",
        "importance_uq_n_folds": 0,
        "importance_mean": np.zeros(n_features, dtype=float),
        "importance_variance": np.zeros(n_features, dtype=float),
        "unstable_feature_indices": np.array([], dtype=int),
        "unstable_threshold": 0.0,
    }

    if X_arr.ndim != 2 or X_arr.shape[0] < 4 or n_features <= 0:
        empty["importance_uq_reason"] = "insufficient_shape"
        return empty

    splits = get_inner_cv_splits(
        X_arr,
        y_arr,
        problem_type,
        int(random_state),
        int(max(2, inner_cv_splits)),
        int(max(1, inner_cv_repeats)),
    )
    if not splits:
        empty["importance_uq_reason"] = "no_cv_splits"
        return empty

    max_fold_count = int(max(1, max_folds))
    fold_importances = []
    for fold_idx, (train_idx, _val_idx) in enumerate(splits[:max_fold_count]):
        X_train = X_arr[train_idx]
        y_train = y_arr[train_idx]
        try:
            if problem_type == "classification":
                if np.unique(y_train).size < 2:
                    continue
                model = make_logistic_regression(
                    solver="lbfgs",
                    penalty="l2",
                    C=1.0,
                    max_iter=int(max(100, model_cv_lr_max_iter)),
                    class_weight="balanced",
                    random_state=int(random_state + fold_idx),
                )
            else:
                model = Ridge(alpha=1.0, random_state=int(random_state + fold_idx))

            model.fit(X_train, y_train)
            coef = np.asarray(getattr(model, "coef_", np.zeros(n_features)), dtype=float)
            if coef.ndim == 2:
                imp = np.mean(np.abs(coef), axis=0)
            else:
                imp = np.abs(coef).ravel()
            if imp.size != n_features:
                padded = np.zeros(n_features, dtype=float)
                upto = int(min(n_features, imp.size))
                padded[:upto] = imp[:upto]
                imp = padded
            fold_importances.append(np.asarray(imp, dtype=float))
        except Exception as exc:
            continue

    if len(fold_importances) < int(max(1, min_cv_folds)):
        empty["importance_uq_reason"] = "insufficient_successful_folds"
        empty["importance_uq_n_folds"] = int(len(fold_importances))
        return empty

    mat = np.vstack(fold_importances)
    imp_mean = np.mean(mat, axis=0)
    if mat.shape[0] > 1:
        imp_var = np.var(mat, axis=0, ddof=1)
    else:
        imp_var = np.zeros(n_features, dtype=float)

    imp_var = np.nan_to_num(np.asarray(imp_var, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    # FIX T-A3-FIX-006: Use coefficient of variation (CV > 1.0) instead of
    # the 50th percentile, which always flags ~50% of features.
    imp_mean_abs = np.abs(np.nan_to_num(np.asarray(imp_mean, dtype=float), nan=0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(imp_mean_abs > 1e-12, np.sqrt(imp_var) / imp_mean_abs, 0.0)
    cv = np.nan_to_num(cv, nan=0.0, posinf=0.0, neginf=0.0)
    threshold = 1.0  # coefficient of variation > 1.0 → "unstable"
    unstable = np.where(cv > threshold)[0].astype(int)

    return {
        "importance_uq_enabled": True,
        "importance_uq_computed": True,
        "importance_uq_reason": "ok",
        "importance_uq_n_folds": int(mat.shape[0]),
        "importance_mean": np.asarray(imp_mean, dtype=float),
        "importance_variance": np.asarray(imp_var, dtype=float),
        "unstable_feature_indices": np.asarray(unstable, dtype=int),
        "unstable_threshold": float(threshold),
    }

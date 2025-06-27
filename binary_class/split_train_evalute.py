# split_train_evalute.py

import numpy as np
from data_preparation import build_preprocessor
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def suppurt_proba(model, X_processed):
    # إذا كان النموذج يدعم الاحتمالات
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_processed)[:, 1]
    else:
        y_prob = None
    return y_prob


def include_regression_fun(model):
    if "regression" in model.__class__.__name__.lower():
        return True
    return False


# تقييم شامل للنموذج باستخدام تقسيم train/test
def train_test_taker(model, X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)

    y_prob = suppurt_proba(model, X_test_processed)
    include_regression = include_regression_fun(model)

    return evaluate_model_performance(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        include_regression=include_regression,
    ) | {"train_test_predictions": y_pred}


def cross_validation_taker(model, X, y, cv=5):
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    # المقاييس المطلوبة
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        scoring["roc_auc"] = "roc_auc"

    # حساب المقاييس لكل طيّة
    results = cross_validate(
        model,
        X_processed,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    include_regression = include_regression_fun(model)
    mean_scores = {
        f"mean_{metric}": results[f"test_{metric}"].mean() for metric in scoring
    }
    all_folds = {
        f"{metric}_per_fold": results[f"test_{metric}"].tolist() for metric in scoring
    }

    # الحصول على التنبؤات والاحتمالات لكل نقطة
    y_pred = cross_val_predict(model, X_processed, y, cv=cv)

    try:
        y_prob = cross_val_predict(
            model, X_processed, y, cv=cv, method="predict_proba"
        )[:, 1]
    except:
        y_prob = None

    include_regression = include_regression_fun(model)
    performance_metrics = evaluate_model_performance(
        y_true=y, y_pred=y_pred, y_prob=y_prob, include_regression=include_regression
    )

    return mean_scores | all_folds


""" | performance_metrics | {
        "cv_predictions": y_pred,
        "cv_probabilities": y_prob,
    }
"""


def stratified_kfold_taker(model, X, y, cv=5):
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # تعريف المقاييس
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        scoring["roc_auc"] = "roc_auc"

    # حساب المقاييس لكل طيّة
    results = cross_validate(
        model,
        X_processed,
        y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
    )

    # المتوسطات
    mean_scores = {
        f"mean_{metric}": results[f"test_{metric}"].mean() for metric in scoring
    }

    # نتائج كل طيّة
    all_folds = {
        f"{metric}_per_fold": results[f"test_{metric}"].tolist() for metric in scoring
    }

    # التنبؤات والاحتمالات لكل عينة
    y_pred = cross_val_predict(model, X_processed, y, cv=skf)

    try:
        y_prob = cross_val_predict(
            model, X_processed, y, cv=skf, method="predict_proba"
        )[:, 1]
    except:
        y_prob = None

    include_regression = include_regression_fun(model)
    performance_metrics = evaluate_model_performance(
        y_true=y, y_pred=y_pred, y_prob=y_prob, include_regression=include_regression
    )

    return mean_scores | all_folds


""" | performance_metrics | {
        "skf_predictions": y_pred,
        "skf_probabilities": y_prob,
    }
"""


def evaluate_model_performance(y_true, y_pred, y_prob=None, include_regression=False):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    # حساب ROC AUC باستخدام الاحتمالات إذا توفرت
    try:
        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        else:
            # fallback في حال لم تتوفر الاحتمالات
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
    except ValueError:
        metrics["roc_auc"] = None

    if include_regression:
        mse = mean_squared_error(y_true, y_pred)
        metrics.update(
            {
                "mae": mean_absolute_error(y_true, y_pred),
                "mse": mse,
                "rmse": np.sqrt(mse),
            }
        )
    return metrics
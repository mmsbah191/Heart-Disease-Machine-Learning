# evaluate_split.py

import numpy as np
from sklearn.model_selection import train_test_split,cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef
)

from data_preparation import build_preprocessor



def suppurt_proba(model,X_processed):
        # إذا كان النموذج يدعم الاحتمالات
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_processed)[:, 1]
    else:
        y_prob = None
    y_prob
    
def include_regression_fun(model):
    if "regression" in model.__class__.__name__.lower():
        return True
    return False

# تقييم شامل للنموذج باستخدام تقسيم train/test
def evaluate_with_train_test(model, X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)

    y_prob=suppurt_proba(model,X_test_processed)
    include_regression=include_regression_fun(model)

    return evaluate_model_performance(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        include_regression=include_regression,
    ) | {"train_test_predictions": y_pred}


# تقييم باستخدام Cross-Validation
def evaluate_with_cross_validation(model, X, y, cv=5):
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    y_pred = cross_val_predict(model, X_processed, y, cv=cv)

    y_prob=suppurt_proba(model,X_processed)
    include_regression=include_regression_fun(model)

    return evaluate_model_performance(
        y_true=y, y_pred=y_pred, y_prob=y_prob, include_regression=include_regression
    ) | {"cv_predictions": y_pred}


# تقييم باستخدام Stratified K-Fold
def evaluate_with_stratified_kfold(model, X, y, cv=5):
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_processed, y, cv=skf)

    y_prob=suppurt_proba(model,X_processed)
    include_regression=include_regression_fun(model)

    return evaluate_model_performance(
        y_true=y, y_pred=y_pred, y_prob=y_prob, include_regression=include_regression
    ) | {"skf_predictions": y_pred}


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

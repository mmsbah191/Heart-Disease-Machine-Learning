# split_train_evalute.py
import numpy as np
from data_preparation import build_preprocessor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    make_scorer,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)


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
        metrics["roc_auc"] = roc_auc_score(
            y_true, y_prob if y_prob is not None else y_pred
        )
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


def suppurt_proba(model, X_processed):
    # إذا كان النموذج يدعم الاحتمالات
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_processed)[:, 1]
    else:
        return None


def include_regression_fun(model):
    return "regression" in model.__class__.__name__.lower()


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
    ) | {
        "predictions": y_pred,
        "probabilities": y_prob,
    }


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
    try:
        y_prob = cross_val_predict(
            model, X_processed, y, cv=cv, method="predict_proba"
        )[:, 1]
    except:
        y_prob = None

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
    # مصفوفة الالتباس هي أداة تحليلية مهمة جدًا في تصنيف البيانات،
    # لأنها توضح لك تفاصيل ما وراء الدقة (accuracy)
    # . وهي خاصة مفيدة في حالة البيانات غير المتوازنة.
    y_pred = cross_val_predict(model, X_processed, y, cv=cv)
    cm = {"confusion_matrix": confusion_matrix(y, y_pred)}

    include_regression = include_regression_fun(model)
    performance_metrics = evaluate_model_performance(
        y_true=y, y_pred=y_pred, y_prob=y_prob, include_regression=include_regression
    )

    y_pred = cross_val_predict(model, X_processed, y, cv=cv)

    return (
        mean_scores
        | cm
        | {
            "all_folds": all_folds,
            "performance_metrics": performance_metrics,
            "predictions": y_pred,
            "probabilities": y_prob,
        }
    )


def stratified_kfold_taker(model, X, y, cv=5):

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    return cross_validation_taker(model, X, y, cv=skf)

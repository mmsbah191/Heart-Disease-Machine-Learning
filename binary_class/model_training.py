# model_training.py

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error
)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef, accuracy_score
)

from data_preparation import get_column_types, build_preprocessor

# تقييم شامل للنموذج باستخدام تقسيم train/test

def evaluate_with_train_test(model, X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)

    return compute_metrics(y_test, y_pred)

# تقييم باستخدام Cross-Validation

def evaluate_with_cross_validation(model, X, y, cv=5):
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    y_pred = cross_val_predict(model, X_processed, y, cv=cv)
    
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_pred),
        "cohen_kappa": cohen_kappa_score(y, y_pred),
        "mcc": matthews_corrcoef(y, y_pred),
        "cv_predictions": y_pred,
    }

# تقييم باستخدام Stratified K-Fold

def evaluate_with_stratified_kfold(model, X, y, cv=5):
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_processed, y, cv=skf)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_pred),
        "cohen_kappa": cohen_kappa_score(y, y_pred),
        "mcc": matthews_corrcoef(y, y_pred),
        "skf_predictions": y_pred,
    }


# حساب المقاييس التفصيلية للأداء

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }

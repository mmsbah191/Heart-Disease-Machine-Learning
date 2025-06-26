import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from data_preparation import load_data, build_preprocessor

# NEW: Import resampling methods from imblearn
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# إعداد البيانات
X, y = load_data()
preprocessor = build_preprocessor(X)

# إعداد التقسيم الطبقي للكروس فاليديشن
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Number of features after preprocessing (for SelectKBest and PCA)
preprocessor.fit(X)
try:
    n_features = preprocessor.transform(X).shape[1]
except Exception:
    n_features = X.shape[1]

# تعريف النماذج والمعاملات
models_and_params = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, solver='liblinear'),
        'params': {
            'resampler': ['passthrough', SMOTE(), RandomOverSampler()],
            'feature_selection': ['passthrough', SelectKBest(f_classif)],
            'feature_selection__k': [5, 8, 'all'],
            'dim_reduction': ['passthrough', PCA()],
            'dim_reduction__n_components': [2, 5, 8, n_features],
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2'],
            'model__class_weight': [None, 'balanced']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {
            'resampler': ['passthrough', SMOTE(), RandomOverSampler()],
            'feature_selection': ['passthrough', SelectKBest(f_classif)],
            'feature_selection__k': [5, 8, 'all'],
            'dim_reduction': ['passthrough', PCA()],
            'dim_reduction__n_components': [2, 5, 8, n_features],
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 3, 5, 10],
            'model__min_samples_split': [2, 5, 10],
            'model__class_weight': [None, 'balanced']
        }
    },
    'SVM': {
        'model': SVC(probability=True),
        'params': {
            'resampler': ['passthrough', SMOTE(), RandomOverSampler()],
            'feature_selection': ['passthrough', SelectKBest(f_classif)],
            'feature_selection__k': [5, 8, 'all'],
            'dim_reduction': ['passthrough', PCA()],
            'dim_reduction__n_components': [2, 5, 8, n_features],
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__kernel': ['rbf', 'linear'],
            'model__class_weight': [None, 'balanced']
        }
    }
}

results = {}

for name, mp in models_and_params.items():
    print(f'\n===== {name} Hyperparameter Tuning with Feature Selection/PCA/Resampling =====')
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('resampler', 'passthrough'),
        ('feature_selection', 'passthrough'),
        ('dim_reduction', 'passthrough'),
        ('model', mp['model'])
    ])
    param_grid = mp['params']
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1, error_score='raise')
    grid.fit(X, y)
    print(f'Best parameters: {grid.best_params_}')
    print(f'Best ROC-AUC: {grid.best_score_:.4f}')
    results[name] = {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'cv_results': grid.cv_results_
    }

# يمكن حفظ النتائج أو طباعتها بشكل مفصل لاحقًا 
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from data_preparation import load_data, build_preprocessor
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import json
import os

# --- Data Preparation ---
def load_data(filepath="../datasets/heart_disease_cleaned.csv"):
    """Load the cleaned heart disease dataset and return features X and target y."""
    df = pd.read_csv(filepath)
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

def build_preprocessor(X):
    """Build a ColumnTransformer for preprocessing categorical and numerical features."""
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    preprocessor = Pipeline([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ])
    return preprocessor

# --- Main Hyperparameter Tuning Logic ---

# Load data and build preprocessor
X, y = load_data()
preprocessor = build_preprocessor(X)

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Fit preprocessor to get number of features after transformation
preprocessor.fit(X)
try:
    n_features = preprocessor.transform(X).shape[1]
except Exception:
    n_features = X.shape[1]

# Dynamically filter k and n_components values for SelectKBest and PCA
k_values = [k for k in [5, 8, 'all'] if k == 'all' or (isinstance(k, int) and k <= n_features)]
n_components_values = [nc for nc in [2, 5, 8, n_features] if isinstance(nc, int) and nc <= n_features]
if n_features not in n_components_values:
    n_components_values.append(n_features)

# Define models and their parameter grids for GridSearchCV
models_and_params = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, solver='liblinear'),
        'params': {
            'resampler': ['passthrough', SMOTE(), RandomOverSampler()],
            'feature_selection': ['passthrough', SelectKBest(f_classif)],
            'feature_selection__k': k_values,
            'dim_reduction': ['passthrough', PCA()],
            'dim_reduction__n_components': n_components_values,
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
            'feature_selection__k': k_values,
            'dim_reduction': ['passthrough', PCA()],
            'dim_reduction__n_components': n_components_values,
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
            'feature_selection__k': k_values,
            'dim_reduction': ['passthrough', PCA()],
            'dim_reduction__n_components': n_components_values,
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__kernel': ['rbf', 'linear'],
            'model__class_weight': [None, 'balanced']
        }
    }
}

results = {}
results_file = 'gridsearch_results.json'
error_log_file = 'gridsearch_errors.log'

for name, mp in models_and_params.items():
    print(f'\n===== {name} Hyperparameter Tuning with Feature Selection/PCA/Resampling =====')
    # Build the pipeline for each model
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('resampler', 'passthrough'),
        ('feature_selection', 'passthrough'),
        ('dim_reduction', 'passthrough'),
        ('model', mp['model'])
    ])
    param_grid = mp['params']
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1, error_score='raise')
    try:
        grid.fit(X, y)
        print(f'Best parameters: {grid.best_params_}')
        print(f'Best ROC-AUC: {grid.best_score_:.4f}')
        results[name] = {
            'best_params': grid.best_params_,
            'best_score': grid.best_score_,
            'cv_results': None
        }
        # Save/update results to JSON after each model
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        all_results[name] = {
            'best_params': grid.best_params_,
            'best_score': grid.best_score_
        }
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    except Exception as e:
        error_msg = f'Error for {name}: {str(e)}\n'
        print(error_msg)
        with open(error_log_file, 'a') as f:
            f.write(error_msg)

# Results can be further processed, printed, or visualized as needed. 
# data_preparation.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# قراءة البيانات من الملف عن طريق load_data()
# وتعريف الأعمدة الفئوية والعدية
# وتحضير البيانات للتدريب


# قراءة الملف المنظف واستخراج X, y
def load_data(filepath="datasets/heart_disease_cleaned.csv"):
    try:
        df = pd.read_csv(filepath)
    except:
        df = pd.read_csv("../" + filepath)
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


# تعريف الأعمدة الفئوية والعددية
def get_column_types(X):
    categorical_columns = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    return categorical_columns, numerical_columns


# بناء ColumnTransformer لمعالجة البيانات
def build_preprocessor(X):
    categorical_columns, numerical_columns = get_column_types(X)
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
            ("num", StandardScaler(), numerical_columns),
        ]
    )

    # لو تبي أسماء الأعمدة للمقارنات
    # preprocessor.fit(X)
    # encoder = preprocessor.named_transformers_['cat']
    # cat_feature_names = encoder.get_feature_names_out(categorical_columns)

    # all_feature_names = list(cat_feature_names) + numerical_columns
    # return preprocessor, all_feature_names

    return preprocessor


# تحضير لتقسيم


# تحضير كامل لل X, y لـ CV أو StratifiedKFold
def prepare_full_data(X, y):
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor


# test
# X, y = load_data()
# X_processed, y, preprocessor = prepare_full_data(X, y)
# print(X_processed.shape)  # dimmision

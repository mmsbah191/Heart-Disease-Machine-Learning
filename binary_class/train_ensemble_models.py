# ensemble_models.py
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from evaluate_split import *
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score,
    matthews_corrcoef, mean_absolute_error, mean_squared_error
)
import numpy as np
from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from data_preparation import build_preprocessor



# تعريف النماذج المختلفة المستخدمة في التصنيف (Ensemble Models)
models = {
        # نموذج Stacking Classifier: يتم دمج العديد من النماذج الأساسية
        # (Base Learners) ويتم استخدام نموذج نهائي للتصنيف.
        # Stacking = التعلم من "تنبؤات" عدة نماذج.
        "Stacking Classifier(knn,nb,dt)": StackingClassifier(
            estimators=[  # النماذج الأساسية
                ("knn", KNeighborsClassifier()),  # نموذج KNN
                ("nb", GaussianNB()),  # نموذج Naive Bayes
                ("dt", DecisionTreeClassifier()),  # نموذج Decision Tree
            ],
            final_estimator=LogisticRegression(),  # النموذج النهائي الذي يدمج نتائج النماذج الأساسية
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        # خفيف وسريع، مناسب لبيانات نصية أو عندما تكون الميزات مستقلة نسبيًا
        "Naive Bayes": GaussianNB(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "SVM (RBF Kernel)": SVC(probability=True),  # RBF kernel هو الافتراضي
        # نموذج Random Forest: نموذج الغابات العشوائية، يعتمد على بناء العديد من الأشجار العشوائية وتجمعها للتصنيف.
        "Random Forest": RandomForestClassifier(),
        # نموذج Extra Trees: مشابه للغابات العشوائية، لكنه يختلف في طريقة بناء الأشجار بشكل أسرع باستخدام طريقة عشوائية أكثر.
        # "Extra Trees": ExtraTreesClassifier(),
        # نموذج AdaBoost: يستخدم تعزيز النماذج الضعيفة عبر تعديل الأوزان لكل نموذج بناءً على أدائه.
        # "AdaBoost": AdaBoostClassifier(),
        # نموذج Gradient Boosting: يعتمد على بناء مجموعة من النماذج الضعيفة بشكل متسلسل وتحسينها في كل مرة.
        "Gradient Boosting": GradientBoostingClassifier(),
        # نموذج Bagging باستخدام Decision Tree: يتم بناء عدة أشجار قرار بشكل متوازٍ (بـ Bagging) وتحسين النتائج عبر دمجها.
        "Bagging (DecisionTree)": BaggingClassifier(estimator=DecisionTreeClassifier()),
        # نموذج Bagging باستخدام KNN: بناء عدة نماذج KNN بشكل متوازٍ لتحسين الأداء.
        "Bagging (KNN)": BaggingClassifier(estimator=KNeighborsClassifier()),
        # # ولأفضل أداء XGBoost أو LightGBM خاصة لو كان عندك بيانات كبيرة أو غير متوازنة
        # "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        # "LightGBM": LGBMClassifier(),
        # نموذج التصويت (Hard Voting): يعتمد على الأغلبية.
        # يعني أنه يتم التصويت من قبل جميع النماذج،
        # ويتم اختيار الفئة التي حصلت على أكبر عدد من الأصوات.
        "Voting (Hard)": VotingClassifier(
            estimators=[  # النماذج التي سيتم استخدامها في التصويت
                ("lr", LogisticRegression(max_iter=1000)),  # نموذج الانحدار اللوجستي
                ("rf", RandomForestClassifier()),  # نموذج الغابات العشوائية
                ("svc", SVC(probability=True)),  # نموذج الـ SVM مع تمكين الاحتمالية
            ],
            voting="hard",  # نوع التصويت (Hard Voting)
        ),
        # نموذج التصويت (Soft Voting): يعتمد على الاحتمالات.
        # يتم أخذ متوسط الاحتمالات
        # التي تُقدمها النماذج المختلفة
        # ويتم اختيار الفئة التي لديها أعلى احتمال.
        "Voting (Soft)": VotingClassifier(
            estimators=[  # النماذج التي سيتم استخدامها في التصويت
                ("lr", LogisticRegression(max_iter=1000)),  # نموذج الانحدار اللوجستي
                ("rf", RandomForestClassifier()),  # نموذج الغابات العشوائية
                ("svc", SVC(probability=True)),  # نموذج الـ SVM مع تمكين الاحتمالية
            ],
            voting="soft",  # نوع التصويت (Soft Voting)
        ),
    }



def evaluate_ensemble_models(X,y):
    ensemble_results = {}
    for name, model in models.items():
        metrics=evaluate_with_train_test(model, X, y)
        metrics=evaluate_with_cross_validation(model, X, y)
        metrics=evaluate_with_stratified_kfold(model, X, y)
        ensemble_results[name] = metrics

    return ensemble_results

"""
النماذج الأساسية (base learners)
تتعلم من البيانات، ثم تُستخدم تنبؤاتها كمدخلات لنموذج آخر (final estimator).
يتم دمج تنبؤات النماذج الأساسية لتكوين تنبؤ أكثر دقة.
الأداء غالبًا يكون أفضل لأن كل نموذج يُغطي نقطة ضعف نموذج آخر.
"""

"""
    Hard Voting يكون أسرع في الحساب لأنه يعتمد فقط على التصويت، بينما 
    Soft Voting يتطلب حساب الاحتمالات وبالتالي قد يكون أبطأ ولكنه عادة يكون أكثر دقة.
    """

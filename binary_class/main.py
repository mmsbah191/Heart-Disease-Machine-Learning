import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from data_preparation import load_data
from train_ensemble_models import evaluate_ensemble_models
from evaluate_split import (evaluate_with_cross_validation,
                            evaluate_with_stratified_kfold,
                            evaluate_with_train_test)


def plot_cv_boxplot(cv_scores_dict):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=list(cv_scores_dict.values()))
    plt.xticks(
        ticks=range(len(cv_scores_dict)),
        labels=list(cv_scores_dict.keys()),
        rotation=45,
    )
    plt.title("Cross-Validation Accuracy Score Distribution")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()


def plot_model_comparison_bar(comparison_df):
    plt.figure(figsize=(10, 6))
    comparison_df.set_index("Model")[["Accuracy", "ROC-AUC"]].plot(kind="bar")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def main():

    
    # تحميل البيانات بعد التنظيف
    X, y = load_data()

    # سنجرب فقط نموذج الانحدار اللوجستي مع max_iter=1000
    model_name = "Logistic Regression"
    model = LogisticRegression(max_iter=1000)

    print(f"\n▶ {model_name} - Train/Test Split Evaluation")
    result = evaluate_with_train_test(model, X, y)
    for metric, res in result.items():
        if isinstance(res, float):
            print(f"{metric}: {res:.4f}")
        else:
            print(f"{metric}:")
            print(res)

    print(f"\n▶ {model_name} - Cross-Validation Evaluation")
    result = evaluate_with_cross_validation(model, X, y)
    for metric, res in result.items():
        if metric != "cv_predictions":
            print(f"{metric}: {res:.4f}")


    print(f"\n▶ {model_name} - Stratified K-Fold Evaluation")
    result = evaluate_with_stratified_kfold(model, X, y)
    for metric, res in result.items():
        if metric != "skf_predictions":
            print(f"{metric}: {res:.4f}")

    print("\n▶ Ensemble Model Comparison")
    ensemble_results = evaluate_ensemble_models(X, y)

    # جمع لكل نموذج للرسم البياني
    cv_scores_all = {}

    for ens_model, metrics in ensemble_results.items():
        print(f"\n{ens_model}:")
        for metric, res in metrics.items():
            if metric == "confusion_matrix":
                print(f"  {metric}:")
                print(res)
                # عرض مصفوفة الارتباك رسمياً
                plot_confusion_matrix(res, ens_model)
            elif res is None:
                print(f"  {metric}: None")
            else:
                print(f"  {metric}: {res:.4f}")



        # تخزين دقة CV للرسم البياني إذا متوفر cv_scores
        # لكن في النسخة الحالية cv_scores غير موجودة، نستخدم 'accuracy' كمقياس
        if "accuracy" in metrics:
            cv_scores_all[ens_model] = np.repeat(
                metrics["accuracy"], 5
            )  # مجرد تقريب لـ boxplot

    if cv_scores_all:
        plot_cv_boxplot(cv_scores_all)




if __name__ == "__main__":
    main()

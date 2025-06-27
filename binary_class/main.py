import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data_preparation import load_data
from sklearn.linear_model import LogisticRegression
from split_train_evalute import (
    cross_validation_taker,
    stratified_kfold_taker,
    train_test_taker,
)
from train_ensemble_models import evaluate_ensemble_models


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
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
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
    result = train_test_taker(model, X, y)
    for metric, res in result.items():
            if isinstance(res, float):
                print(f"{metric}: {res:.4f}")
            elif metric == "confusion_matrix":
                print(f"  {metric}:")
                print(res)
                # plot_confusion_matrix(res, model)
                pass

    print(f"\n▶ {model_name} - Cross-Validation Evaluation")
    result = cross_validation_taker(model, X, y)
    for metric, res in result.items():
            if isinstance(res, float):
                print(f"{metric}: {res:.4f}")
            elif metric == "confusion_matrix":
                print(f"  {metric}:")
                print(res)
                # plot_confusion_matrix(res, model)
                pass

    print(f"\n▶ {model_name} - Stratified K-Fold Evaluation")
    result = stratified_kfold_taker(model, X, y)
    for metric, res in result.items():
            if isinstance(res, float):
                print(f"{metric}: {res:.4f}")
            elif metric == "confusion_matrix":
                print(f"  {metric}:")
                print(res)
                # plot_confusion_matrix(res, model)
                pass

    print("\n▶ Ensemble Model Comparison")
    ensemble_results = evaluate_ensemble_models(X, y)

    # جمع لكل نموذج للرسم البياني
    cv_scores_all = {}

    for ens_model, evaluate_metrics in ensemble_results.items():
        print(f"\n{ens_model}:")
        for split_method, metrics in evaluate_metrics.items():
            print(f"\n🔸 {split_method.upper()} Evaluation:")
            for metric, res in metrics.items():
                if metric in (
                    "all_folds",
                    "predictions",
                    "performance_metrics",
                    "probabilities",
                ):
                    continue
                if isinstance(res, float):
                    print(f"{metric}: {res:.4f}")
                elif metric == "confusion_matrix":
                    print(f"  {metric}:")
                    print(res)
                    # plot_confusion_matrix(res, ens_model)


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
    print("finish main")

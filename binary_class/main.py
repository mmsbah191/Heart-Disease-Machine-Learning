import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data_preparation import load_data
from sklearn.linear_model import LogisticRegression
from split_train_evalute import (cross_validation_taker,
                                 stratified_kfold_taker, train_test_taker)
from train_ensemble_models import evaluate_ensemble_models
from visualizations import (  # تأكد من أن الدوال موجودة أو منسوخة هنا
    plot_accuracy_vs_rocauc, plot_confusion_matrix, plot_cv_boxplot,
    plot_metric_distribution, plot_model_comparison_bar, plot_model_heatmap,plot_top_models)


def main():
    X, y = load_data()
    model_name = "Logistic Regression"
    model = LogisticRegression(max_iter=1000)

    print(f"\n▶ {model_name} - Train/Test Split Evaluation")
    result = train_test_taker(model, X, y)
    for per_metric, value in result.items():
        if isinstance(value, float):
            print(f"{per_metric}: {value:.4f}")
        elif per_metric == "confusion_matrix":
            print(f"  {per_metric}:")
            print(value)
            plot_confusion_matrix(value,f"{model_name} - Train/Test Split Evaluation")
    

    print(f"\n▶ {model_name} - Cross-Validation Evaluation")
    result = cross_validation_taker(model, X, y)
    for per_metric, value in result.items():
        if isinstance(value, float):
            print(f"{per_metric}: {value:.4f}")
        elif per_metric == "confusion_matrix":
            print(f"  {per_metric}:")
            print(value)
            plot_confusion_matrix(value,f"{model_name} - Cross-Validation Evaluation")

    print(f"\n▶ {model_name} - Stratified K-Fold Evaluation")
    result = stratified_kfold_taker(model, X, y)
    for per_metric, value in result.items():
        if isinstance(value, float):
            print(f"{per_metric}: {value:.4f}")
        elif per_metric == "confusion_matrix":
            print(f"  {per_metric}:")
            print(value)
            plot_confusion_matrix(value,f"{model_name} - Stratified K-Fold Evaluation")

    print("\n▶ Ensemble Model Comparison")
    ensemble_results = evaluate_ensemble_models(X, y)

    cv_scores_all = {}
    comparison_data_all = []
    comparison_data_cv = []
    comparison_data_skv = []

    for model_name, model_result in ensemble_results.items():
        print(f"\n{model_name}:")
        for split_method, metrics in model_result.items():
            model_split_name = model_name + " " + split_method.upper()
            print(f"\n▶ {model_split_name} Evaluation:")
            for per_metric, value in metrics.items():
                if per_metric in ("all_folds", "predictions", "performance_metrics", "probabilities"):
                    continue
                if isinstance(value, float):
                    print(f"{per_metric}: {value:.4f}")
                elif per_metric == "confusion_matrix":
                    print(f"  {per_metric}:")
                    print(value)
                    # يمكنك تفعيل عرض المصفوفة إذا رغبت:
                    # plot_confusion_matrix(value, model_split_name)

            if "accuracy" in metrics:
                cv_scores_all[model_split_name] = np.repeat(metrics["accuracy"], 5)
            elif "mean_accuracy" in metrics:
                cv_scores_all[model_split_name] = np.repeat(metrics["mean_accuracy"], 5)

            accuracy = metrics.get("accuracy", metrics.get("mean_accuracy", 0))
            roc_auc = metrics.get("roc_auc", metrics.get("mean_roc_auc", 0))
            f1 = metrics.get("f1_score", metrics.get("mean_f1", 0))

            row = {"Model": model_split_name, "Accuracy": accuracy, "ROC-AUC": roc_auc, "F1": f1}

            comparison_data_all.append(row)
            if split_method.upper() == "CROSS_VAL":
                comparison_data_cv.append(row)
            elif split_method.upper() == "STRATIFIED_KFOLD":
                comparison_data_skv.append(row)

    # 📊 Full comparison
    comparison_df = pd.DataFrame(comparison_data_all)
    print("\n📊 Model Comparison Table (All):")
    print(comparison_df)
    plot_model_comparison_bar(comparison_df, ["Accuracy", "F1", "ROC-AUC"], title_suffix="(All Ensemble Models)")
    plot_accuracy_vs_rocauc(comparison_df)
    plot_model_heatmap(comparison_df)

    # 🔍 أفضل نموذج
    if not comparison_df.empty:
        best_f1_model = comparison_df.sort_values("F1", ascending=False).iloc[0]
        best_auc_model = comparison_df.sort_values("ROC-AUC", ascending=False).iloc[0]
        print(f"\n Best Model by F1 Score: {best_f1_model['Model']} (F1={best_f1_model['F1']:.4f})")
        print(f" Best Model by ROC-AUC: {best_auc_model['Model']} (ROC-AUC={best_auc_model['ROC-AUC']:.4f})")
    plot_top_models(comparison_df, metric="F1", top_n=5)
    plot_top_models(comparison_df, metric="ROC-AUC", top_n=5)
    
    # 📊 CV Only
    comparison_df_cv = pd.DataFrame(comparison_data_cv)
    print("\n📊 Model Comparison Table (Cross-Validation):")
    print(comparison_df_cv)
    plot_model_comparison_bar(comparison_df_cv, ["Accuracy", "F1", "ROC-AUC"], title_suffix="(CV)")
    plot_accuracy_vs_rocauc(comparison_df_cv)
    
        # 🔍 أفضل نموذج
    if not comparison_df.empty:
        best_f1_model = comparison_df.sort_values("F1", ascending=False).iloc[0]
        best_auc_model = comparison_df.sort_values("ROC-AUC", ascending=False).iloc[0]
        print(f"\n Best Model by F1 Score: {best_f1_model['Model']} (F1={best_f1_model['F1']:.4f})")
        print(f" Best Model by ROC-AUC: {best_auc_model['Model']} (ROC-AUC={best_auc_model['ROC-AUC']:.4f})")
    plot_top_models(comparison_df, metric="F1", top_n=5)
    plot_top_models(comparison_df, metric="ROC-AUC", top_n=5)

    # 📊 SKFold Only
    comparison_df_skv = pd.DataFrame(comparison_data_skv)
    print("\n📊 Model Comparison Table (Stratified K-Fold):")
    print(comparison_df_skv)
    plot_model_comparison_bar(comparison_df_skv, ["Accuracy", "F1", "ROC-AUC"], title_suffix="(SKFold)")
    plot_accuracy_vs_rocauc(comparison_df_skv)
    
        # 🔍 أفضل نموذج
    if not comparison_df.empty:
        best_f1_model = comparison_df.sort_values("F1", ascending=False).iloc[0]
        best_auc_model = comparison_df.sort_values("ROC-AUC", ascending=False).iloc[0]
        print(f"\n Best Model by F1 Score: {best_f1_model['Model']} (F1={best_f1_model['F1']:.4f})")
        print(f" Best Model by ROC-AUC: {best_auc_model['Model']} (ROC-AUC={best_auc_model['ROC-AUC']:.4f})")
    plot_top_models(comparison_df, metric="F1", top_n=5)
    plot_top_models(comparison_df, metric="ROC-AUC", top_n=5)

    # 🎯 Boxplot لكل النماذج
    plot_cv_boxplot(cv_scores_all)

if __name__ == "__main__":
    main()
    print("finish main")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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


def plot_cv_boxplot(cv_scores_dict):
    if not cv_scores_dict:
        print(" No CV scores available to plot.")
        return

    scores_df = pd.DataFrame(
        {model: scores for model, scores in cv_scores_dict.items()}
    )
    plt.figure(figsize=(max(12, len(scores_df.columns) * 0.6), 6))
    sns.boxplot(data=scores_df)
    plt.xticks(rotation=45, ha="right")
    plt.title("Cross-Validation Accuracy Score Distribution", fontsize=16)
    plt.ylabel("Accuracy", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_model_comparison_bar(name, comparison_df):
    if comparison_df.empty:
        print(" No data to plot in model comparison.")
        return

    num_models = comparison_df.shape[0]
    width = max(12, num_models * 0.6)
    # plt.figure(figsize=(width, 6))
    comparison_df.set_index("Model")[["Accuracy", "ROC-AUC"]].plot(
        kind="bar", figsize=(width, 6)
    )
    plt.title("Model Performance Comparison " + name, fontsize=16)
    plt.ylabel("Score", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()


# رسم جميع المقاييس لكل نموذج
def plot_model_comparison_bar(comparison_df, metrics, title_suffix=""):
    if comparison_df.empty:
        print(" No data to plot.")
        return

    width = max(12, len(comparison_df["Model"]) * 0.6)
    comparison_df.set_index("Model")[metrics].plot(kind="bar", figsize=(width, 6))
    plt.title(f"Model {title_suffix} Comparison", fontsize=16)
    plt.ylabel("Score", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()


# رسم Heatmap للمقاييس بين النماذج
def plot_model_heatmap(comparison_df):
    if comparison_df.empty:
        return
    metrics_df = comparison_df.set_index("Model").copy()
    plt.figure(figsize=(12, 6))
    sns.heatmap(metrics_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Performance Metrics Heatmap")
    plt.tight_layout()
    plt.show()


# رسم scatter plot لتوضيح العلاقة بين دقة و ROC-AUC
def plot_accuracy_vs_rocauc(comparison_df):
    if comparison_df.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=comparison_df, x="Accuracy", y="ROC-AUC", hue="Model", s=100)
    plt.title("Accuracy vs. ROC-AUC")
    plt.xlabel("Accuracy")
    plt.ylabel("ROC-AUC")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metric_distribution(metric_values, metric_name):
    plt.figure(figsize=(8, 4))
    sns.histplot(metric_values, kde=True)
    plt.title(f"Distribution of {metric_name}")
    plt.xlabel(metric_name)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_top_models(comparison_df, metric="F1", top_n=5):
    if comparison_df.empty or metric not in comparison_df.columns:
        print(f"⚠️ No data or '{metric}' not found.")
        return
    top_models = comparison_df.sort_values(metric, ascending=False).head(top_n)
    plt.figure(figsize=(8, 5))
    colors = ["green"] + ["gray"] * (len(top_models) - 1)
    sns.barplot(x=metric, y="Model", data=top_models, hue="Model", palette=colors, dodge=False, legend=False)
    plt.title(f"Top {top_n} Models by {metric}", fontsize=14)
    plt.xlabel(metric)
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()

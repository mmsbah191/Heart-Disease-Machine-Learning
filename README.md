# Early-Detection-of-Heart-Disease-Using-Machine-Learning


📄 Final Status Report
**Heart Disease Classification Project**

---

🏷 **Project Title**
**Classification of Heart Disease Using Machine Learning and Ensemble Models**

👤 **Student Name**
**Mohamed Mesbah & Aymen Mildi**

🧾 **Course**
**CS461 - Machine Learning**

---

🟩 **Weekly Progress Summary**

✅ **This Week’s Accomplishments**

- Completed implementation of data preprocessing using:
  - One-Hot Encoding
  - Standard Scaling
- Built evaluation modules using:
  - Hold-out Train/Test Split
  - K-Fold Cross-Validation
  - Stratified K-Fold Cross-Validation
- Integrated a wide range of ensemble models, including:
  - `StackingClassifier`
  - Voting (Hard & Soft)
  - Bagging (with KNN, DecisionTree)
  - Random Forest, Gradient Boosting
- Extracted and reported multiple performance metrics:
  - Accuracy, Precision, Recall, F1-score
  - ROC-AUC, MCC, Cohen-Kappa
  - Confusion Matrix
- Visualizations:
  - Boxplot: Cross-validation score distribution
  - Bar chart: Accuracy vs. ROC-AUC across models
  - Heatmaps: Confusion matrices

---

🕒 **Are You On Schedule?**
✅ Yes – The project is on track. All planned features for the mid-project milestone were implemented successfully.

---

📍 **Milestones Met**☑ Data Cleaning & Preprocessing☑ Modular code structure:

- `data_preparation.py`
- `model_training.py`
- `ensemble_models.py`
  ☑ Evaluation of multiple individual and ensemble classifiers
  ☑ Mid-project reporting and visualization

---

📊 **Midway Results Summary**

- **Best individual model (so far):**
  - Logistic Regression — Accuracy ≈ **87%**, ROC-AUC ≈ **89%**
- **Best ensemble model:**
  - Voting (Soft) or Gradient Boosting — Accuracy ≈ **88–90%**, ROC-AUC ≈ **91%**
- **Confusion matrix heatmaps:**
  - Show more true positives than false negatives — favorable for medical diagnostic applications.

---

📌 **Upcoming Milestones & Plans**

🔬 **Experiments in Progress**

- Hyperparameter tuning via `GridSearchCV`
- Evaluation using external datasets (e.g., Cleveland dataset)
- Plotting ROC and Precision-Recall (PR) curves
- Testing under imbalanced data scenarios

🧪 **Planned Experiments**

- Compare ensemble robustness across varying `StratifiedKFold` seeds
- Use **SHAP** or permutation importance for interpretability
- Detect overfitting: Compare training vs. validation metrics

---

📆 **Timeline**


| **Week** | **Focus Areas**                                |
| -------- | ---------------------------------------------- |
| Week 6   | Hyperparameter tuning, extra visualizations    |
| Week 7   | Final testing, interpretability, full write-up |

---

📘 **References**

- Scikit-learn Documentation
  🔗 https://scikit-learn.org/
- UCI Heart Disease Dataset
  🔗 https://archive.ics.uci.edu/ml/datasets/heart+Disease
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*

---

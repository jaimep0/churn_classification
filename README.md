## Project Overview: Telecom Customer Churn Prediction

**Interconnect**, a telecommunications operator, aims to reduce customer churn by implementing a predictive machine learning system. By proactively identifying at-risk subscribers, the business can deploy targeted retention campaigns, customized promotions, and specialized plan offers before cancellation occurs.

* **Primary Objective:** Build and evaluate binary classification models to predict imminent customer churn (`1 = Yes`, `0 = No`).
* **Evaluation Strategy:** **AUC-ROC** serves as the primary evaluation metric, with **Recall (Class 1)** prioritized as the key secondary metric to minimize missed churners.

---

## Model Evaluation & Performance Breakdown

The top-performing model achieved strong predictive capability across all core metrics on the holdout test set:

| Metric | Score | Performance Context |
| :--- | :--- | :--- |
| **Accuracy** | **0.89** | Overall correct classification rate across the test dataset ($N = 705$). |
| **Recall (Class 1)** | **0.77** | Correctly identifies **77% of all actual churners** (~8 out of 10), enabling timely intervention by the marketing team. |
| **Recall (Class 0)** | **0.93** | Correctly identifies **93% of non-churning customers**, keeping false alarms down to just 7%. |
| **Precision (Class 1)**| **0.80** | Ensures that 80% of flagged customers are true churn risks, preventing unnecessary promotional spending. |

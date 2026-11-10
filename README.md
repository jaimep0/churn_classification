Telecom operator Interconnect aims to reduce customer churn by implementing a system to proactively identify users at risk of canceling their service, allowing the team to engage them with targeted promotions and specialized plans.

Goal: Develop and evaluate predictive models to determine whether a customer will cancel their service in the near future (Yes/No).

Evaluation Metrics: Primary evaluation will be based on AUC-ROC, with Recall serving as a key secondary metric.


Recall (Class 1 – Churn): 0.77
Measures the model's ability to catch actual churners. Out of all customers who actually canceled, the model identified 77% (nearly 8 out of 10). This enables the marketing team to proactively target them with retention offers.Recall (Class 0 – Non-Churn): 0.93
Measures the model's coverage of actual non-churners. Out of all customers who stayed, 93% were correctly identified as non-churners, meaning only 7% were falsely flagged as at-risk (false alarms).Accuracy: 0.89
Overall, the model correctly classifies 89% of all customers in the test dataset. Given the class imbalance (~73% non-churn vs ~27% churn), the solid performance across both Precision (0.80) and Recall (0.77) for Class 1 confirms the high accuracy isn't just driven by majority-class bias.

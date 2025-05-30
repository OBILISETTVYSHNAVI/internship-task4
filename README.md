# internship-task4
Internship

Logistic Regression Classifier – Breast Cancer Detection

 🎯 Project Aim

The aim of this project is to build a binary classification model using **Logistic Regression** to accurately predict whether a tumor is **malignant** or **benign** based on diagnostic measurements. This is a foundational project that highlights how machine learning can support early detection of breast cancer — one of the most critical applications of AI in healthcare.

By the end of this project, you'll understand how Logistic Regression works, how to evaluate classification models using standard metrics, and how to interpret the output using visual tools like the sigmoid curve and ROC curve.



 🧠 Introduction to Logistic Regression

**Logistic Regression** is a fundamental statistical technique used for binary classification problems. Unlike linear regression that predicts continuous values, logistic regression predicts the **probability** of a data point belonging to a particular class (e.g., tumor being malignant or benign).

The key component of logistic regression is the **sigmoid function**, which maps any real-valued number into a value between 0 and 1:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

This probability is compared against a threshold (typically 0.5) to classify the outcome.





 ✅ Task Guidelines (Internship Mini-Guide)

This project strictly follows the step-by-step workflow outlined in the internship mini guide:

1. **Choose a Binary Classification Dataset:**
   - Used the Breast Cancer Wisconsin dataset from Kaggle.

2. **Train/Test Split:**
   - The dataset is split into 80% training and 20% testing sets.

3. **Standardize Features:**
   - Applied feature scaling using `StandardScaler` to normalize input features for better model performance.

4. **Fit Logistic Regression Model:**
   - Trained a logistic regression model using `scikit-learn`.

5. **Evaluate the Model:**
   - Metrics used:
     - **Confusion Matrix**
     - **Precision**
     - **Recall**
     - **F1 Score**
     - **ROC-AUC Score**
     - **ROC Curve Visualization**

6. **Tune Threshold and Explain Sigmoid Function:**
   - Plotted the sigmoid function to show how probabilities are mapped to binary decisions.
   - Discussed decision boundary tuning by changing the threshold from 0.5 if needed.



 📊 Evaluation Metrics & Visualizations

To evaluate the performance of the logistic regression model, the following metrics and visual tools are used:

 🔷 1. Confusion Matrix
A table that describes the performance of the classification model by comparing predicted and actual values.

|                    | Predicted Benign (0) | Predicted Malignant (1) |
|--------------------|----------------------|--------------------------|
| **Actual Benign (0)**    | True Negative (TN)     | False Positive (FP)         |
| **Actual Malignant (1)** | False Negative (FN)    | True Positive (TP)          |

✅ Helps identify where the model is making errors — especially useful in medical diagnosis where false negatives are critical.

 🔷 2. Precision
\[
\{Precision} = {TP}/{TP + FP}
\]

Measures the accuracy of positive predictions. High precision means fewer false positives.

 🔷 3. Recall (Sensitivity)
\[
\{Recall} = \{TP}/{TP + FN}
\]

Measures the ability to find all relevant cases (true positives). High recall is important in medical diagnostics.

 🔷 4. F1 Score
\[
F1 = 2*\{\{Precision}* \{Recall}}/{{Precision} + {Recall}}
\]

Balances precision and recall.

 🔷 5. ROC-AUC Score
Represents the area under the ROC curve. Closer to 1.0 means better classification performance.



 📈 Visual Outputs

- Confusion Matrix Heatmap
- ROC Curve
- Sigmoid Function Plot


## 📌 Summary

This project demonstrates how Logistic Regression, a simple yet powerful algorithm, can be applied to solve real-world binary classification problems. Through this task, we also explored key evaluation metrics and the role of thresholds in tuning classification sensitivity.

By mastering this workflow, you're gaining practical machine learning experience with real medical data — an essential foundation for further ML and AI applications.

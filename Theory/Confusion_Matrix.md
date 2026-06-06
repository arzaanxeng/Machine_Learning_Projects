# Understanding the Confusion Matrix

When evaluating a machine learning classification model, relying solely on Accuracy can be misleading—especially with imbalanced datasets. To get a true picture of how your model is performing, we use a **Confusion Matrix**.

A Confusion Matrix is a tabular layout that compares a model's **Predicted Labels** against the **Actual (True) Labels**.

---

## The Core Structure (2x2 Matrix)

For a binary classification problem (e.g., Classifying an email as *Spam* or *Not Spam*), the matrix consists of four quadrants:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | **True Positive (TP)** <br> *Correctly predicted Positive.* | **False Negative (FN)** <br> *Missed it! Predicted Negative.* (Type II Error) |
| **Actual Negative** | **False Positive (FP)** <br> *False Alarm! Predicted Positive.* (Type I Error) | **True Negative (TN)** <br> *Correctly predicted Negative.* |

### Breaking Down the Four Quadrants:
* **True Positive (TP):** You predicted positive, and it’s true. *(e.g., Predicted sick, and they are actually sick).*
* **True Negative (TN):** You predicted negative, and it’s true. *(e.g., Predicted healthy, and they are actually healthy).*
* **False Positive (FP):** You predicted positive, but it's false. *(e.g., Predicted sick, but they are actually healthy).*
* **False Negative (FN):** You predicted negative, but it's false. *(e.g., Predicted healthy, but they are actually sick).*

---

## Key Evaluation Metrics

From these four values, we can calculate the core metrics used to judge classification models:

### 1. Accuracy
The overall proportion of correct predictions.
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

### 2. Precision
Out of all the examples the model *predicted* as positive, how many were actually positive? (Crucial when False Positives are expensive, like Spam Filters).
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

### 3. Recall (Sensitivity)
Out of all the *actual* positive examples, how many did the model manage to find? (Crucial when False Negatives are dangerous, like Medical Diagnoses).
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

### 4. F1-Score
The harmonic mean of Precision and Recall. It gives a balanced measure on imbalanced datasets.
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## Python Implementation

Here is how to generate and plot a confusion matrix using `scikit-learn` and `seaborn`:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Example Ground Truth and Predictions
y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1, 1, 1, 1, 0]

# Generate the matrix
cm = confusion_matrix(y_true, y_pred)

# Plotting using Seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])

plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix Breakdown')
plt.show()

# Print text-based classification report
print(classification_report(y_true, y_pred))

# Breast Cancer Classification with Logistic Regression

A simple, step-by-step machine learning workflow that trains a Logistic Regression model to classify breast cancer tumor samples as benign or malignant, using `scikit-learn`.

## Overview

This project reads tabular tumor-cell measurements from a CSV file, trains a Logistic Regression classifier, evaluates it with a confusion matrix, and validates its performance using k-Fold Cross Validation.

## Prerequisites

- Python 3.x
- The following Python libraries:
  - `pandas`
  - `scikit-learn`

Install them with:

```bash
pip install pandas scikit-learn
```

## Dataset

- File: `breast_cancer.csv`
- The dataset contains cell-nuclei measurements (e.g. Clump Thickness, Bland Chromatin, Normal Nucleoli, Mitoses, etc.) with a final `Class` column used as the prediction target.
- Place `breast_cancer.csv` in the same directory as your script/notebook before running the steps below.

## Original Data from here: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
<img width="1028" height="627" alt="Screenshot 2026-09-18 134142" src="https://github.com/user-attachments/assets/538eaad1-1ac2-450f-894a-e14d5d83c647" />
<img width="1656" height="748" alt="Screenshot 2026-09-18 140055" src="https://github.com/user-attachments/assets/1b9b8120-8869-437a-970f-bc73b296c94e" />
## Steps

### 1. Import the libraries

```python
import pandas as pd  # pandas is a python library to import datasets in various formats e.g. csv, JSON etc
```
## <img width="1412" height="461" alt="Screenshot 2026-09-18 134002" src="https://github.com/user-attachments/assets/99817c88-82b1-4cb7-8df4-603466322922" />

### 2. Import the dataset

```python
dataset = pd.read_csv('breast_cancer.csv')  # import the data using the read_csv function from the pandas library
X = dataset.iloc[:, 1:-1].values  # iloc is index location — gets data from all rows and from the Clump Thickness column to the Mitoses column
y = dataset.iloc[:, -1].values    # gets the target/label column (Class)
```

### 3. Split the dataset into the Training set and Test set

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
<img width="1028" height="627" alt="Screenshot 2026-09-18 134142" src="https://github.com/user-attachments/assets/02cb430a-53f2-4249-adcd-e795b461cd22" />

### 4. Train the Logistic Regression model on the Training set

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
```

### 5. Predict the Test set results

```python
y_pred = classifier.predict(X_test)
```

### 6. Make the Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

Example output:

```
[[84  3]
 [ 3 47]]
```

Accuracy from the confusion matrix:

```python
(84 + 47) / (84 + 47 + 3 + 3)
```

```
0.9562043795620438
```

### 7. Compute accuracy with k-Fold Cross Validation

```python
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
```

Example output:

```
Accuracy: 96.70 %
Standard Deviation: 1.97 %
```

## Results Summary

| Metric | Value |
|---|---|
| Test set accuracy (from confusion matrix) | 95.62% |
| 10-Fold Cross Validation accuracy | 96.70% |
| Cross Validation standard deviation | 1.97% |

## Notes

- `random_state=0` is used throughout for reproducibility.
- The confusion matrix indicates 84 true negatives, 47 true positives, and 3 false positives / 3 false negatives.
- k-Fold Cross Validation (`cv=10`) is used to get a more robust estimate of model performance than a single train/test split.

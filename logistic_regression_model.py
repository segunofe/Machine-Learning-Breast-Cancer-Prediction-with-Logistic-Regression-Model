#Logistic Regression

##Importing the libraries
"""

import pandas as pd # pandas is a python library used to import dataset in various formats e.g csv, JSON etc

"""## Importing the dataset"""

dataset = pd.read_csv('breast_cancer.csv') # import the data using the read_csv function.

# X contains the independent variables (features)
X = dataset.iloc[:, 1:-1].values # iloc is index location # This line selects all rows, columns from index 1 up to (but not including) the last column

# y contains the dependent variable (target)
y = dataset.iloc[:, -1].values # Selects all rows from the last column

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""## Training the Logistic Regression model on the Training set"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

"""## Predicting the Test set results"""

y_pred = classifier.predict(X_test)

"""## Making the Confusion Matrix"""

from sklearn.metrics import confusion_matrix # Confusing matrix will tell us how many correct predictions our model did and incorrect predictions it did
cm = confusion_matrix(y_test, y_pred) # vector of the real result, ground truth (y_test) we want to compare to our prediction vector (y_pred)
print(cm)

(84+47)/(84+47+3+3)

"""## Computing the accuracy with k-Fold Cross Validation"""

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

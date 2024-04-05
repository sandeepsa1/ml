from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import time

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Introduce noise into the data
np.random.seed(42)
X_noisy = X + np.random.normal(0, 0.55, X.shape)

# Split the noisy dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42)

# Initialize classifiers
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train Random Forest classifier
start_time = time.time()
rf_clf.fit(X_train, y_train)
rf_train_time = time.time() - start_time
rf_pred = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Train AdaBoost classifier
start_time = time.time()
ada_clf.fit(X_train, y_train)
ada_train_time = time.time() - start_time
ada_pred = ada_clf.predict(X_test)
ada_accuracy = accuracy_score(y_test, ada_pred)

# Train Gradient Boosting classifier
start_time = time.time()
gb_clf.fit(X_train, y_train)
gb_train_time = time.time() - start_time
gb_pred = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)

# Make predictions
rf_pred = rf_clf.predict(X_test)
ada_pred = ada_clf.predict(X_test)
gb_pred = gb_clf.predict(X_test)

# Calculate accuracies
rf_accuracy = accuracy_score(y_test, rf_pred)
ada_accuracy = accuracy_score(y_test, ada_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)

# Print accuracies
print("Random Forest Accuracy:", rf_accuracy)
print("AdaBoost Accuracy:", ada_accuracy)
print("Gradient Boosting Accuracy:", gb_accuracy)

print("Random Forest Training Time:", rf_train_time, "seconds")
print("AdaBoost Training Time:", ada_train_time, "seconds")
print("Gradient Boosting Training Time:", gb_train_time, "seconds")

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import time

# Load the Wine Quality dataset
data = load_wine()
X = data.data
y = data.target

# Introduce noise into the data
np.random.seed(42)
X_noisy = X + np.random.normal(0, 1.5, X.shape)

# Split the noisy dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42)

# Initialize classifiers
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# List of classifiers
classifiers = [(rf_clf, "Random Forest"), (ada_clf, "AdaBoost"), (gb_clf, "Gradient Boosting")]

# Train and evaluate each classifier
for clf, name in classifiers:
    print(f"Training {name}...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"Evaluating {name}...")
    start_time = time.time()
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    eval_time = time.time() - start_time
    
    print(f"{name} - Accuracy: {accuracy:.4f}, Training Time: {train_time:.2f} seconds, Evaluation Time: {eval_time:.2f} seconds")
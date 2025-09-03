# This model to predict whether a voice is male or female based on its acoustic features using Support Vector Machines (SVM) and RandomizedSearchCV.

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load and clean data
data = pd.read_csv("C://Users//akjee//Documents//ML//voice.csv")
print("Data Size:", data.shape)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
print(data.head())

# One-hot encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split features and target
x = data.iloc[:, :-1]  # Independent variables
y = data.iloc[:, -1]   # Dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
print(f"X Train shape is :{X_train.shape}")
print(f"X Test shape is :{X_test.shape}")
print(f"Y Train shape is :{y_train.shape}")
print(f"Y Test shape is :{y_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using RandomizedSearchCV
param_dist = {
    'C': np.logspace(-2, 2, 10),
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

random_search = RandomizedSearchCV(
    SVC(),
    param_distributions=param_dist,
    n_iter=100,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit model
random_search.fit(X_train_scaled, y_train)

# Best parameters
print("Best Parameters:", random_search.best_params_)
best_svc = random_search.best_estimator_
print(best_svc)

# Predictions
y_pred = best_svc.predict(X_test_scaled)
print("Predictions:", y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_svc.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification report and accuracy
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Model Accuracy:", random_search.score(X_test_scaled, y_test))

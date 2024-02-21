from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("data/predicted_dataset_probs.csv")

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(numeric_cols)

Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the numeric columns, then transform them
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


X = df[["anger","brain dysfunction (forget)","emptiness","hopelessness","loneliness","sadness","suicide intent","worthlessness"]]
y = df["pandemic_period"]


# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Define the hyperparameters
hyperparameters = {
    "Logistic Regression": {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    },
    "Decision Tree": {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 5, 10]
    },
    "Random Forest": {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 5, 10]
    }
}

best_model = None
best_score = 0


# Tune the classifiers
for name, clf in classifiers.items():
    grid = GridSearchCV(clf, hyperparameters[name], cv=5)
    grid.fit(X_train, y_train)
    print(f"{name} Best Parameters: {grid.best_params_}")
    print(f"{name} Best Score: {grid.best_score_}")

    # Evaluate on the validation set
    val_predictions = grid.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"{name} Validation Accuracy: {val_accuracy}")

    if val_accuracy > best_score:
        best_score = val_accuracy
        best_model = grid
        best_model_name = name

# Evaluate on the test set
test_predictions = best_model.predict(X_test)
print(f"Best Model Test Accuracy: {best_model_name} has {accuracy_score(y_test, test_predictions)}")
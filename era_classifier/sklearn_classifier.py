from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from CovidDataset import load_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
from joblib import dump

    
X, y = load_data("experimental")

# Specifying test ratio
test_ratio = 0.15
# Splitting the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)

# Creating a pipeline to test different ML models using grid search
pipe = Pipeline([('clf', 'passthrough')])

# Defining the ML models , along with their chosen hyperparameters, to be tested
parameters = [
    {
        'clf' : [LinearSVC()],
        'clf__C' : [1, 10, 100]
    },
    {
        'clf':[LogisticRegression()],
        'clf__C':[1, 10, 100],
        
    },
    {
        'clf':[RandomForestClassifier()],
        'clf__n_estimators':[100, 200, 400],
        'clf__criterion':["gini", "entropy", "log_loss"]
    },
    {
        'clf' : [AdaBoostClassifier()],
        'clf__n_estimators' : [100, 200, 400],
        'clf__learning_rate' : [0.01, 0.1, 1]
    }
]


# Defining the GridSearchCV object
opt= GridSearchCV(pipe, parameters, scoring='accuracy', verbose=3, return_train_score=True, cv=10, error_score="raise")
# Performing a GridSearchCV on the training dataset with 10-Fold Cross Validation and Accuracy as the evaluation metric
opt.fit(X_train, y_train)
# Saving the grid search result into a CSV file
results = pd.DataFrame.from_dict(opt.cv_results_)
results.to_csv("/results/sklearn/gridsearch_result.csv", index=False)
# Retrieving the best model from grid search and saving it in a file
print(f"Best Parameters : {opt.best_params_} with Average Validation Accuracy of {opt.best_score_}")
best_model = opt.best_estimator_
dump(best_model, "/results/sklearn/best_model.joblib")
# Using the best model to make predictions on the test dataset, and then computing the accuracy
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Saving the testing accuracy of the best model into a CSV file
with open("results/sklearn/test_result.csv", "w") as f:
    f.write(f"test_accuracy\n{accuracy}")
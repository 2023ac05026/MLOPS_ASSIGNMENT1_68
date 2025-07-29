
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend


# Enable MLflow autologging
mlflow.autolog()

with mlflow.start_run():
    # Load Data
    df = pd.read_csv("Data/processed/processed_iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split Data (added random_state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Start of GridSearchCV Modifications ---

    # 1. Define the parameter grid to search
    # These are the hyperparameters that GridSearchCV will test.
    param_grid = {
        'C': [0.1, 1, 10, 100],          # Regularization strength
        'solver': ['liblinear', 'saga'], # Solvers that support both l1 and l2
        'penalty': ['l1', 'l2']          # Regularization penalty type
    }

    # 2. Instantiate the base model
    # max_iter is increased to ensure convergence for some solvers.
    model = LogisticRegression(max_iter=500)

    # 3. Set up GridSearchCV
    # This will test all combinations of parameters in param_grid using 5-fold cross-validation.
    # n_jobs=-1 uses all available CPU cores to speed up the search.
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

    # 4. Fit GridSearchCV to find the best model
    print("Running GridSearchCV...")
    grid_search.fit(X_train, y_train)

    # 5. Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)

    # --- End of GridSearchCV Modifications ---


    # --- MLflow Logging ---

    # Log the best parameters found by the grid search
    mlflow.log_params(best_params)

    # Log the cross-validation score of the best model
    mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)

    # Evaluate the best model on the held-out test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_set_accuracy", test_accuracy)

    # Log the final, tuned model
    mlflow.sklearn.log_model(best_model, "logistic_regression_tuned_model")

    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print("Model saved to run:", mlflow.active_run().info.run_id)

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier # Changed import
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend


# Enable MLflow autologging
mlflow.autolog()

with mlflow.start_run():

# Load Data
    processed_data_dir = os.path.join("Data", "processed")
    processed_data_path = os.path.join(processed_data_dir, "processed_iris.csv")
    df = pd.read_csv(processed_data_path)
    df = pd.read_csv(processed_data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split Data (added random_state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Start of GridSearchCV Modifications for Random Forest ---

    # 1. Define the parameter grid to search for Random Forest
    # These are common and impactful hyperparameters for this algorithm.
    param_grid = {
        'n_estimators': [50, 100, 200],      # Number of trees in the forest
        'max_depth': [None, 10, 20],         # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],     # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4]        # Minimum number of samples required at a leaf node
    }

    # 2. Instantiate the base model
    model = RandomForestClassifier(random_state=42) # Changed model

    # 3. Set up GridSearchCV
    # The process is identical, just with the new model and param_grid.
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

    # 4. Fit GridSearchCV to find the best model
    print("Running GridSearchCV for RandomForestClassifier...")
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

    # Log the final, tuned model with a descriptive name
    mlflow.sklearn.log_model(best_model, "random_forest_tuned_model") # Changed model name

    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print("Model saved to run:", mlflow.active_run().info.run_id)
#!/usr/bin/env python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
from pathlib import Path

def train_rf_model(data_path):
    """
    Loads data, trains a Random Forest model using GridSearchCV,
    and logs the results with MLflow.

    Args:
        data_path (str): The path to the processed CSV data file.

    Returns:
        tuple: A tuple containing the best trained model and its test accuracy.
    """
    # Enable MLflow autologging for this training session
    mlflow.autolog()

    with mlflow.start_run(log_system_metrics=True) as run:
        # Load Data
        df = pd.read_csv(data_path)
        X = df.drop("target", axis=1)
        y = df["target"]

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Instantiate and run GridSearchCV
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
        print("Running GridSearchCV for RandomForestClassifier...")
        grid_search.fit(X_train, y_train)

        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print("Best parameters found: ", best_params)

        # Evaluate the best model
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Manually log key metrics and params (autolog handles the rest)
        mlflow.log_params(best_params)
        mlflow.log_metric("test_set_accuracy", test_accuracy)
        mlflow.log_inputs()
        mlflow.log_trace()
        mlflow.log_artifacts("model_randomforest")
        
        print(f"Test Set Accuracy: {test_accuracy:.4f}")
        print("Model saved to run:", run.info.run_id)
        
        return best_model, test_accuracy

if __name__ == '__main__':
    # This block allows the script to be run directly
    # Set up MLflow tracking URI
    if "MLFLOW_TRACKING_URI" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        mlruns_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlruns'))
        local_uri = Path(mlruns_dir).as_uri()
        mlflow.set_tracking_uri(local_uri)

    # Define the data path
    processed_data_path = os.path.join("Data", "processed", "processed_iris.csv")
    
    # Run the training
    train_rf_model(data_path=processed_data_path)
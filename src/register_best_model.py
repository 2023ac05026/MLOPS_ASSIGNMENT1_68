#!/usr/bin/env python
import mlflow
import os
import pandas as pd
from pathlib import Path
from mlflow.tracking import MlflowClient

# --- Configuration ---
# The name of the experiment you want to search.
EXPERIMENT_NAME = "Default" 

# The metric to use for ranking models.
METRIC_TO_OPTIMIZE = "metrics.test_set_accuracy"

# The name you want to use for the model in the Model Registry.
REGISTERED_MODEL_NAME = "iris-classifier"

# --- Search for the Best Run ---

print(f"Searching for the best run in experiment: '{EXPERIMENT_NAME}'")

# --- Correctly Set MLflow Tracking URI ---
if "MLFLOW_TRACKING_URI" in os.environ:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    print(f"MLflow tracking URI set from environment variable: {os.environ['MLFLOW_TRACKING_URI']}")
else:
    mlruns_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlruns'))
    local_uri = Path(mlruns_dir).as_uri()
    mlflow.set_tracking_uri(local_uri)
    print(f"MLflow tracking URI set to local path: {local_uri}")

# Initialize the MLflow client
client = MlflowClient()

# Get the experiment by name to find its ID
try:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
except AttributeError:
    print(f"Experiment '{EXPERIMENT_NAME}' not found.")
    exit()


# Search all runs in the specified experiment, ordered by our metric
runs_df = mlflow.search_runs(
    experiment_ids=[experiment_id], 
    order_by=[f"{METRIC_TO_OPTIMIZE} DESC"]
)

if runs_df.empty:
    print("No runs found in this experiment.")
else:
    # The best run is the first one in the sorted DataFrame
    best_run = runs_df.iloc[0]
    best_run_id = best_run["run_id"]
    best_run_accuracy = best_run[METRIC_TO_OPTIMIZE]
    
    # Construct the model URI from the best run
    model_uri = f"runs:/{best_run_id}/model" # Using a generic 'model' path is common

    print(f"\n--- Best Model Found ---")
    print(f"Run ID: {best_run_id}")
    print(f"Test Set Accuracy: {best_run_accuracy:.4f}")
    print(f"Model URI: {model_uri}")

    # --- Register the Best Model ---
    
    print(f"\nRegistering model '{REGISTERED_MODEL_NAME}'...")

    try:
        # The core function to register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME
        )
        print("\n✅ Model successfully registered!")
        print(f"Name: {model_version.name}")
        print(f"Version: {model_version.version}")
        print(f"Stage: {model_version.current_stage}")

    except Exception as e:
        print(f"Error registering model: {e}")
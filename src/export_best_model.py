#!/usr/bin/env python

import mlflow
import joblib
import os
from pathlib import Path

print("--- Starting Model Export ---")

# --- Configuration ---
REGISTERED_MODEL_NAME = "iris-classifier"
MODEL_STAGE = "None"

'''if "MLFLOW_TRACKING_URI" in os.environ:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# --- Set MLflow Tracking URI ---
mlruns_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlruns'))
MLFLOW_TRACKING_URI = Path(mlruns_dir).as_uri()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")'''

# This logic allows the script to work both locally and in a container.
if "MLFLOW_TRACKING_URI" in os.environ:
    # Use the URI from the environment variable if it exists (for Docker)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    print(f"MLflow tracking URI set from environment variable: {os.environ['MLFLOW_TRACKING_URI']}")
else:
    # Otherwise, fall back to the local mlruns directory (for local development)
    mlruns_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlruns'))
    local_uri = Path(mlruns_dir).as_uri()
    mlflow.set_tracking_uri(local_uri)
    print(f"MLflow tracking URI set to local path: {local_uri}")

# --- Load Model from MLflow ---
try:
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    print(f"Loading model from URI: {model_uri}")
    # Load the scikit-learn model flavor
    loaded_model = mlflow.sklearn.load_model(model_uri)
    print("Model loaded successfully from MLflow.")
except Exception as e:
    print(f"Error loading model from MLflow: {e}")
    exit(1)

# --- Export Model to Joblib File ---
EXPORT_DIR = os.path.join("app", "models")
EXPORT_FILE_NAME = "model.joblib"

os.makedirs(EXPORT_DIR, exist_ok=True)
export_path = os.path.join(EXPORT_DIR, EXPORT_FILE_NAME)

try:
    joblib.dump(loaded_model, export_path)
    print(f"Model successfully exported to: {export_path}")
except Exception as e:
    print(f"Error exporting model: {e}")
    exit(1)

print("--- Model Export Finished ---")

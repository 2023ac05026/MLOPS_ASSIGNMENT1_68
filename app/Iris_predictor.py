# app/Iris_predictor.py
import os
import mlflow
import pandas as pd
from flask import Flask, request, jsonify, render_template
from pathlib import Path
import joblib

# --- Create a robust, OS-independent file URI for MLflow ---
# Get the absolute path to the mlruns directory, which is one level up from the app directory
mlruns_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlruns'))
# Convert the path to a file URI that works on all operating systems
MLFLOW_TRACKING_URI = Path(mlruns_dir).as_uri()
# Set the MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- Model Loading ---
# Name of the registered model in the MLflow Model Registry
REGISTERED_MODEL_NAME = "iris-classifier"
MODEL_STAGE = "None"  # Or "Staging", "Production" if you use stages

print(f"Loading model '{REGISTERED_MODEL_NAME}' from tracking URI: {MLFLOW_TRACKING_URI}")
try:
    # Use the 'models:/' URI to load the latest version for the given stage
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit if the model fails to load, as the app cannot function
    exit()
# --- Load the standalone model file ---
# The path is relative to the app's location inside the container

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the model file
MODEL_PATH = os.path.join(script_dir, 'models', 'model.joblib')

try:
    loaded_model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define the target names for interpreting the prediction results
TARGET_NAMES = ['setosa', 'versicolor', 'virginica']

# Initialize the Flask application
app = Flask(__name__)

# --- Routes ---
@app.route('/')
def home():
    """Render the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the web form."""
    try:
        # Get data from the POST request form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Create a pandas DataFrame with the correct column names for the model
        input_data = pd.DataFrame({
            'sepal length (cm)': [sepal_length],
            'sepal width (cm)': [sepal_width],
            'petal length (cm)': [petal_length],
            'petal width (cm)': [petal_width]
        })

        # Make a prediction using the loaded model
        prediction_code = loaded_model.predict(input_data)[0]
        predicted_species = TARGET_NAMES[prediction_code]

        # Return the result to the HTML page for display
        return render_template('index.html',
                               prediction_text=f'Predicted Species: {predicted_species}')

    except Exception as e:
        # Handle any errors during prediction
        return render_template('index.html',
                               prediction_text=f'Error: {str(e)}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Handle API prediction requests (for programmatic use)."""
    try:
        # Get JSON data from the request body
        data = request.get_json(force=True)
        input_data = pd.DataFrame(data)

        # Make predictions
        prediction = loaded_model.predict(input_data)
        
        # Convert numeric predictions to species names
        output = [TARGET_NAMES[p] for p in prediction]
        
        return jsonify(output)

    except Exception as e:
        return jsonify({'error': str(e)})

# This block allows the script to be run directly with `python Iris_predictor.py`
# for local testing. It is ignored when run with `flask run`.
if __name__ == '__main__':
    # Run the app on host 0.0.0.0 to be accessible from outside the container
    app.run(host='0.0.0.0', port=5001, debug=True)

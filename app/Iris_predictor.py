import os
import mlflow
import pandas as pd
from flask import Flask, request, jsonify, render_template
from pathlib import Path # Import the Path object

# --- FIX: Use pathlib to create a robust, OS-independent file URI ---

# Get the absolute path to the mlruns directory
mlruns_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlruns'))

# Convert the absolute path to a URI
# This works correctly on both Windows and Linux/macOS
MLFLOW_TRACKING_URI = Path(mlruns_dir).as_uri()

# Set the MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Name of the registered model
REGISTERED_MODEL_NAME = "iris-classifier"
MODEL_STAGE = "None" # Or "Staging", "Production" if you use stages

# Load the model from the Model Registry
print(f"Loading model '{REGISTERED_MODEL_NAME}' from {MLFLOW_TRACKING_URI}")
try:
    # Use the 'models:/' URI to load the latest version for the given stage
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit if model fails to load
    exit()

# Define the target names for interpreting the prediction
TARGET_NAMES = ['setosa', 'versicolor', 'virginica']

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """Render the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get data from the POST request form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # --- FIX: Use column names that match the model's schema ---
        # Create a DataFrame for the model
        input_data = pd.DataFrame({
            'sepal length (cm)': [sepal_length],
            'sepal width (cm)': [sepal_width],
            'petal length (cm)': [petal_length],
            'petal width (cm)': [petal_width]
        })

        # Make a prediction
        prediction_code = loaded_model.predict(input_data)[0]
        predicted_species = TARGET_NAMES[prediction_code]

        # Return the result to the HTML page
        return render_template('index.html', 
                               prediction_text=f'Predicted Species: {predicted_species}')

    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f'Error: {str(e)}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Handle API prediction requests (for programmatic use)."""
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)
        
        # --- FIX: Ensure the DataFrame created from JSON uses correct column names ---
        # The input JSON keys should match this expected format
        input_data = pd.DataFrame(data)

        # Make prediction
        prediction = loaded_model.predict(input_data)
        
        # Convert prediction to a list of species names
        output = [TARGET_NAMES[p] for p in prediction]
        
        return jsonify(output)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the app on host 0.0.0.0 to be accessible from outside the container
    app.run(host='0.0.0.0', port=5001, debug=True)
# app/Iris_predictor.py
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, g
import joblib
import psutil
from pydantic import BaseModel, Field, ValidationError

from collections import deque
import time

try:
    # This works when running as a package (e.g., with 'flask run' or in Docker)
    from . import database, metrics
except ImportError:
    # This works when running the script directly (e.g., 'python app/Iris_predictor.py')
    import database
    import metrics

# --- Pydantic Models for Input Validation ---
class IrisFeatures(BaseModel):
    """
    Pydantic model for validating the input features for Iris prediction.
    Ensures all fields are floats and are greater than zero.
    """
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

# --- Application Setup ---
app = Flask(__name__)
# Register the database functions with the Flask app instance
database.init_app(app)

# Register the metrics blueprint with the main app
metrics.init_metrics(app)
# Create a deque to store the last 100 response times
response_times = deque(maxlen=100)
# --- Model Loading ---
# The path is relative to the app's location
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, 'models', 'model.joblib')

try:
    loaded_model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # In a real scenario, you'd want more robust error handling
    # For now, we exit if the model can't be loaded.
    loaded_model = None
    # exit() # Commented out to allow app to run even if model fails to load

# Define the target names for interpreting the prediction results
TARGET_NAMES = ['setosa', 'versicolor', 'virginica']

# --- Request Timing Hooks ---
@app.before_request
def before_request_timing():
    """Record the start time of a request."""
    g.start_time = time.monotonic()

@app.after_request
def after_request_timing(response):
    """Calculate request duration and add it to our list."""
    if 'start_time' in g:
        duration_ms = (time.monotonic() - g.start_time) * 1000
        response_times.append(duration_ms)
    return response

# --- Routes ---
@app.route('/')
def home():
    print("Running first-request initialization...")
    database.init_db()
    print("Database initialization complete.")
    """Render the home page with the input form."""
    return render_template('index.html')

@app.route('/logs_json')
def view_logs_json():
    """An endpoint to view the prediction logs from the database as JSON."""
    logs = database.query_logs()
    return jsonify(logs)

@app.route('/view_logs')
def view_logs_html():
    """Renders a webpage to display the prediction logs in a table."""
    logs = database.query_logs()
    return render_template('logs.html', logs=logs)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the web form."""
    if loaded_model is None:
        return render_template('index.html', prediction_text='Error: Model is not loaded.')

    try:
        # Validate the form data using the Pydantic model
        validated_data = IrisFeatures(**request.form)
        
        # Create a DataFrame from the validated data with the correct column names
        input_data = pd.DataFrame({
            'sepal length (cm)': [validated_data.sepal_length],
            'sepal width (cm)': [validated_data.sepal_width],
            'petal length (cm)': [validated_data.petal_length],
            'petal width (cm)': [validated_data.petal_width]
        })

        prediction_code = loaded_model.predict(input_data)[0]
        predicted_species = TARGET_NAMES[prediction_code]

        # Log the successful prediction to the database
        database.log_prediction('form', validated_data.model_dump(), predicted_species, status='SUCCESS')

        return render_template('index.html',
                               prediction_text=f'Predicted Species: {predicted_species}')

    except ValidationError as e:
        # Log the validation error to the database
        database.log_prediction('form', request.form.to_dict(), {'error': e.errors()}, status='VALIDATION_ERROR')
        # Return a user-friendly error message for the form
        error_messages = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
        return render_template('index.html',
                               prediction_text=f'Input Validation Error: {", ".join(error_messages)}')

    except Exception as e:
        # Log other errors to the database (e.g., model prediction failed)
        database.log_prediction('form', request.form.to_dict(), {'error': str(e)}, status='ERROR')
        return render_template('index.html',
                               prediction_text=f'Error: {str(e)}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Handle API prediction requests."""
    if loaded_model is None:
         return jsonify({'error': 'Model is not loaded'}), 500

    data = request.get_json(force=True)
    
    try:
        if isinstance(data, list):
            validated_data_list = [IrisFeatures(**item) for item in data]
            input_df_dict = [item.model_dump() for item in validated_data_list]
        else:
            validated_data = IrisFeatures(**data)
            input_df_dict = [validated_data.model_dump()]

        # Create DataFrame from the Pydantic models' data and rename columns
        input_data = pd.DataFrame(input_df_dict).rename(columns={
            'sepal_length': 'sepal length (cm)',
            'sepal_width': 'sepal width (cm)',
            'petal_length': 'petal length (cm)',
            'petal_width': 'petal width (cm)'
        })
        
        prediction = loaded_model.predict(input_data)
        output = [TARGET_NAMES[p] for p in prediction]
        
        # Log the successful API prediction to the database
        database.log_prediction('api', data, output, status='SUCCESS')
        
        return jsonify(output)

    except ValidationError as e:
        # Log the validation error and return a 400 Bad Request with details
        database.log_prediction('api', data, {'error': e.errors()}, status='VALIDATION_ERROR')
        return jsonify({'error': e.errors()}), 400
        
    except Exception as e:
        # Log the general API error to the database
        database.log_prediction('api', data, {'error': str(e)}, status='ERROR')
        return jsonify({'error': str(e)}), 500
        
'''@app.route('/metrics')
def metrics_json():
    """Returns system metrics as a JSON object."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    return jsonify({
        'cpu_percent': cpu_percent,
        'memory': {
            'total': memory_info.total,
            'available': memory_info.available,
            'percent': memory_info.percent,
            'used': memory_info.used,
            'free': memory_info.free
        },
        'disk': {
            'total': disk_info.total,
            'used': disk_info.used,
            'free': disk_info.free,
            'percent': disk_info.percent
        },
        'avg_response_time_ms': avg_response_time
    })'''

@app.route('/metrics_ui')
def metrics_ui():
    """Renders a webpage to display the system metrics."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    metrics_data = {
        'cpu_percent': cpu_percent,
        'memory': memory_info,
        'disk': disk_info,
        'avg_response_time_ms': avg_response_time
    }
    return render_template('metrics.html', metrics=metrics_data)

@app.route('/view_metrics_history')
def view_metrics_history():
    """Renders a webpage to display the historical metrics from the database."""
    metrics = database.query_metrics_history()
    return render_template('metrics_history.html', metrics=metrics)

# --- Main execution block ---
if __name__ == '__main__':
    # Initialize the database if it doesn't exist when running the app directly
    with app.app_context():
        database.init_db()
    app.run(host='0.0.0.0', port=5001, debug=True)
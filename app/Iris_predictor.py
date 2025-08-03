# app/Iris_predictor.py
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, g
import joblib
import psutil
import database # Import the new database module
from collections import deque
import time


# --- Application Setup ---
app = Flask(__name__)
# Register the database functions with the Flask app instance
database.init_app(app)

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

    input_features = {
        'sepal_length': request.form.get('sepal_length'),
        'sepal_width': request.form.get('sepal_width'),
        'petal_length': request.form.get('petal_length'),
        'petal_width': request.form.get('petal_width')
    }

    try:
        # Convert to float for the model
        numeric_features = {k: float(v) for k, v in input_features.items()}
        
        input_data = pd.DataFrame({
            'sepal length (cm)': [numeric_features['sepal_length']],
            'sepal width (cm)': [numeric_features['sepal_width']],
            'petal length (cm)': [numeric_features['petal_length']],
            'petal width (cm)': [numeric_features['petal_width']]
        })

        prediction_code = loaded_model.predict(input_data)[0]
        predicted_species = TARGET_NAMES[prediction_code]

        # Log the successful prediction to the database
        database.log_prediction('form', input_features, predicted_species, status='SUCCESS')

        return render_template('index.html',
                               prediction_text=f'Predicted Species: {predicted_species}')

    except Exception as e:
        # Log the error to the database
        database.log_prediction('form', input_features, {'error': str(e)}, status='ERROR')
        return render_template('index.html',
                               prediction_text=f'Error: {str(e)}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Handle API prediction requests."""
    if loaded_model is None:
         return jsonify({'error': 'Model is not loaded'}), 500

    data = request.get_json(force=True)
    try:
        input_data = pd.DataFrame(data)
        prediction = loaded_model.predict(input_data)
        output = [TARGET_NAMES[p] for p in prediction]
        
        # Log the successful API prediction to the database
        database.log_prediction('api', data, output, status='SUCCESS')
        
        return jsonify(output)

    except Exception as e:
        # Log the API error to the database
        database.log_prediction('api', data, {'error': str(e)}, status='ERROR')
        return jsonify({'error': str(e)})
@app.route('/metrics')
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
    })

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
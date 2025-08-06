# app/metrics.py
import time
import psutil
from collections import deque
from flask import Blueprint, g, jsonify
from prometheus_client import generate_latest, Gauge, Counter, Histogram

# --- Prometheus Metrics Definitions ---
# Counter: A cumulative metric that represents a single monotonically increasing counter whose value can only increase or be reset to zero on restart.
REQUEST_COUNT = Counter(
    'app_requests_total', # Metric name
    'Total number of requests to the application' # Metric description
)
# Counter: Tracks the total number of errors encountered during prediction.
PREDICTION_ERRORS_COUNT = Counter(
    'app_prediction_errors_total',
    'Total number of prediction errors'
)
# Gauge: A metric that represents a single numerical value that can arbitrarily go up and down.
CPU_USAGE_GAUGE = Gauge(
    'app_cpu_percent',
    'Current CPU usage percentage'
)
MEMORY_USAGE_GAUGE = Gauge(
    'app_memory_percent',
    'Current memory usage percentage'
)
DISK_USAGE_GAUGE = Gauge(
    'app_disk_percent',
    'Current disk usage percentage'
)
AVG_RESPONSE_TIME_GAUGE = Gauge(
    'app_avg_response_time_ms',
    'Average response time in milliseconds'
)
# Histogram: Samples observations (e.g., request durations) and counts them in configurable buckets.
REQUEST_LATENCY_HISTOGRAM = Histogram(
    'app_request_latency_seconds',
    'Request latency in seconds',
    # Define custom buckets for more granular latency analysis
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, float('inf'))
)

# Create a deque to store the last 100 response times for internal average calculation
# This is used for the AVG_RESPONSE_TIME_GAUGE and the /metrics_ui endpoint.
response_times_deque = deque(maxlen=100)

# Create a Flask Blueprint for metrics endpoints and hooks.
# Blueprints help organize Flask applications into smaller, reusable components.
metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.before_app_request
def before_request_timing():
    """
    Flask hook executed before each request to the application.
    Records the start time of the request and increments the total request counter.
    `g.start_time` stores the start time in Flask's global request context.
    """
    g.start_time = time.monotonic() # High-resolution monotonic clock for accurate timing
    REQUEST_COUNT.inc() # Increment the Prometheus counter for total requests

@metrics_bp.after_app_request
def after_request_timing(response):
    """
    Flask hook executed after each request to the application.
    Calculates the request duration, adds it to the deque, and updates Prometheus gauges/histograms.
    """
    if 'start_time' in g: # Check if start_time was recorded (it should be by before_request_timing)
        duration_seconds = time.monotonic() - g.start_time # Calculate duration in seconds
        duration_ms = duration_seconds * 1000 # Convert to milliseconds
        response_times_deque.append(duration_ms) # Add to deque for rolling average
        
        # Update Prometheus Gauge for average response time using the deque's average
        AVG_RESPONSE_TIME_GAUGE.set(sum(response_times_deque) / len(response_times_deque) if response_times_deque else 0)
        # Observe the request latency in the histogram for bucketed distribution
        REQUEST_LATENCY_HISTOGRAM.observe(duration_seconds)
    return response

@metrics_bp.route('/metrics')
def prometheus_metrics():
    """
    Endpoint for Prometheus to scrape.
    Updates system-level gauges before returning the metrics in Prometheus exposition format.
    """
    # Get current system metrics using psutil (interval=None for non-blocking immediate value)
    cpu_percent = psutil.cpu_percent(interval=None) 
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')

    # Set the Prometheus Gauge values with current system stats
    CPU_USAGE_GAUGE.set(cpu_percent)
    MEMORY_USAGE_GAUGE.set(memory_info.percent)
    DISK_USAGE_GAUGE.set(disk_info.percent)

    # generate_latest() returns the metrics in Prometheus text exposition format
    # The Content-Type header is crucial for Prometheus to correctly parse the data.
    return generate_latest(), 200, {'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'}

@metrics_bp.route('/system_metrics_json')
def system_metrics_json():
    """
    An internal endpoint to return system metrics as a JSON object.
    This can be used by a UI (like /metrics_ui) or for debugging.
    """
    cpu_percent = psutil.cpu_percent(interval=1) # Use interval=1 for a sampled CPU usage
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    avg_response_time = sum(response_times_deque) / len(response_times_deque) if response_times_deque else 0
    
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

def increment_prediction_errors():
    """
    A helper function to allow the main application to increment the prediction error counter.
    This abstracts away the direct Prometheus client interaction from Iris_predictor.py.
    """
    PREDICTION_ERRORS_COUNT.inc()

def init_metrics(app):
    """
    Initializes the metrics blueprint by registering it with the main Flask application.
    This makes the /metrics and /system_metrics_json routes available and
    activates the before/after request hooks.
    """
    app.register_blueprint(metrics_bp)

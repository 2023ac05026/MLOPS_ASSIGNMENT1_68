import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
# --- Mocking Setup ---
# 1. Create a mock model object that will be used in all tests.
mock_model_instance = MagicMock()

# 2. This is the corrected patch. We target 'joblib.load' directly.
#    When `app.Iris_predictor` is imported, its call to `joblib.load()` will be
#    intercepted by our patch.
with patch('joblib.load', return_value=mock_model_instance):
    from app.Iris_predictor import app

# 3. Create a test client using a pytest fixture. This setup runs once per test function.
@pytest.fixture
def client():
    # Reset the mock before each test to ensure test isolation
    mock_model_instance.reset_mock()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# --- Tests ---

def test_home_route(client):
    """
    Tests if the home page ('/') loads correctly.
    """
    response = client.get('/')
    assert response.status_code == 200
    assert b"Iris Species Predictor" in response.data

def test_predict_endpoint(client):
    """
    Tests the /predict endpoint with valid form data.
    """
    # Configure the mock to return class '1' (versicolor) for this test
    mock_model_instance.predict.return_value = np.array([1])
    test_data = {
        'sepal_length': 5.1,
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2
    }
    response = client.post('/predict', data=test_data)

    assert response.status_code == 200
    mock_model_instance.predict.assert_called_once()
    assert b'Predicted Species: versicolor' in response.data

def test_predict_endpoint_with_invalid_input(client):
    """
    Tests the /predict endpoint with non-numeric (invalid) form data.
    """
    test_data = {
        'sepal_length': 'invalid_data',
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2
    }
    response = client.post('/predict', data=test_data)

    assert response.status_code == 200
    # The model's predict method should not be called with invalid input
    mock_model_instance.predict.assert_not_called()
    assert b'Input Validation Error: sepal_length: Input should be a valid number, unable to parse string as a number' in response.data

def test_api_predict_endpoint(client):
    """
    Tests the /api/predict endpoint with a single valid JSON object.
    """
    # Configure the mock to return class '1' (versicolor) for this test
    mock_model_instance.predict.return_value = np.array([1])
    # The API now expects keys to match the Pydantic model
    test_data = {
        'sepal_length': 5.1,
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2
    }
    response = client.post('/api/predict', json=test_data)
    json_data = response.get_json()

    assert response.status_code == 200
    mock_model_instance.predict.assert_called_once()
    assert json_data == ['versicolor']

def test_api_predict_with_multiple_inputs(client):
    """
    Tests the /api/predict endpoint with a batch of multiple JSON objects.
    """
    # Configure the mock to return predictions for two inputs: 'setosa' (0) and 'virginica' (2)
    mock_model_instance.predict.return_value = np.array([0, 2])
    # The API now expects a list of dictionaries with keys that match the Pydantic model
    test_data = [
        {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        },
        {
            'sepal_length': 7.0,
            'sepal_width': 3.2,
            'petal_length': 4.7,
            'petal_width': 1.4
        }
    ]
    response = client.post('/api/predict', json=test_data)
    json_data = response.get_json()

    assert response.status_code == 200
    mock_model_instance.predict.assert_called_once()
    assert json_data == ['setosa', 'virginica']
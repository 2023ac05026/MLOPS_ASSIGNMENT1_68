# tests/test_iris_predictor.py
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# We import the app here. The model loading will be patched during the tests.
from app.Iris_predictor import app

@pytest.fixture
def client():
    """A pytest fixture to create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_model():
    """A pytest fixture that creates a mock of the loaded ML model."""
    mock_model_instance = MagicMock()
    # Configure its `predict` method to always return class 1 ('versicolor')
    mock_model_instance.predict.return_value = np.array([1])
    return mock_model_instance

@patch('app.Iris_predictor.joblib.load')
def test_home_page_loads(mock_joblib_load, client):
    """
    Test Case 1: Verifies that the home page ('/') loads successfully.
    """
    response = client.get('/')
    assert response.status_code == 200
    assert b"Iris Species Predictor" in response.data

@patch('app.Iris_predictor.loaded_model')
def test_successful_html_prediction(mock_loaded_model, client, mock_model):
    """
    Test Case 2: Simulates a successful form submission to the /predict endpoint.
    """
    # Replace the app's actual model with our mock model for this test
    mock_loaded_model.predict.return_value = mock_model.predict.return_value

    response = client.post('/predict', data={
        'sepal_length': '5.1',
        'sepal_width': '3.5',
        'petal_length': '1.4',
        'petal_width': '0.2'
    })
    assert response.status_code == 200
    # The mock model returns class 1, which is 'versicolor' (lowercase)
    assert b"Predicted Species: versicolor" in response.data

@patch('app.Iris_predictor.joblib.load')
def test_invalid_data_html_prediction(mock_joblib_load, client):
    """
    Test Case 3: Simulates a form submission with invalid (non-numeric) data.
    """
    response = client.post('/predict', data={
        'sepal_length': 'invalid',
        'sepal_width': '3.5',
        'petal_length': '1.4',
        'petal_width': '0.2'
    })
    assert response.status_code == 200
    assert b"Error:" in response.data

@patch('app.Iris_predictor.loaded_model')
def test_successful_api_prediction(mock_loaded_model, client, mock_model):
    """
    Test Case 4: Tests the /api/predict endpoint with valid JSON data.
    """
    # Replace the app's actual model with our mock model for this test
    mock_loaded_model.predict.return_value = mock_model.predict.return_value

    json_data = [{'sepal length (cm)': 5.1, 'sepal width (cm)': 3.5, 'petal length (cm)': 1.4, 'petal width (cm)': 0.2}]
    response = client.post('/api/predict', json=json_data)
    assert response.status_code == 200
    assert response.get_json() == ['versicolor']

@patch('app.Iris_predictor.joblib.load')
def test_api_prediction_with_bad_data(mock_joblib_load, client):
    """
    Test Case 5: Tests the API endpoint with malformed JSON to ensure it handles errors.
    """
    json_data = [{'wrong_column': 1.0}]
    response = client.post('/api/predict', json=json_data)
    assert 'error' in response.get_json()
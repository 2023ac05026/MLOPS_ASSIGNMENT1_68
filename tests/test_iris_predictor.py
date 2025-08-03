# tests/test_iris_predictor.py
import pytest
import os
from unittest.mock import patch, MagicMock
import numpy as np

# Import the 'app' object from your Flask application file
# The conftest.py file ensures that the 'app' directory is in the Python path
from app.Iris_predictor import app

@pytest.fixture
def client():
    """A pytest fixture to create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_model():
    """A pytest fixture that mocks the loaded model."""
    # Create a mock model object that has a `predict` method
    mock_model = MagicMock()
    # Configure the predict method to return a predictable value (e.g., class 1, which is 'versicolor')
    mock_model.predict.return_value = np.array([1])
    return mock_model

def test_home_page_loads(client):
    """
    Test Case 1: Verifies that the home page ('/') loads successfully.
    """
    response = client.get('/')
    # Assert that the HTTP status code is 200 (OK)
    assert response.status_code == 200
    # Assert that the response contains the title of the page
    assert b"Iris Species Predictor" in response.data

@patch('app.Iris_predictor.loaded_model')
def test_successful_html_prediction(mock_loaded_model, client, mock_model):
    """
    Test Case 2: Simulates a successful form submission to the /predict endpoint.
    """
    # Replace the actual loaded_model with our mock_model for this test
    mock_loaded_model.predict = mock_model.predict

    # Simulate sending form data
    response = client.post('/predict', data={
        'sepal_length': '5.1',
        'sepal_width': '3.5',
        'petal_length': '1.4',
        'petal_width': '0.2'
    })

    assert response.status_code == 200
    # Check that the response contains the predicted species name ('versicolor' corresponds to class 1)
    assert b"Predicted Species: versicolor" in response.data

def test_invalid_data_html_prediction(client):
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
    # The application should catch the error and display an error message
    assert b"Error:" in response.data

@patch('app.Iris_predictor.loaded_model')
def test_successful_api_prediction(mock_loaded_model, client, mock_model):
    """
    Test Case 4: Tests the /api/predict endpoint with valid JSON data.
    """
    mock_loaded_model.predict = mock_model.predict

    # Define the JSON payload to send to the API
    json_data = [{
        'sepal length (cm)': 5.1,
        'sepal width (cm)': 3.5,
        'petal length (cm)': 1.4,
        'petal width (cm)': 0.2
    }]

    response = client.post('/api/predict', json=json_data)

    assert response.status_code == 200
    assert response.content_type == 'application/json'
    # Check that the JSON response is a list containing the predicted species
    assert response.get_json() == ['versicolor']

@patch('app.Iris_predictor.joblib.load')
def test_model_loading_failure(mock_joblib_load, client):
    """
    Test Case 5: Simulates a failure to load the model file on startup.
    This test is more advanced and checks how the app handles a critical error.
    """
    # Configure the mock to raise a FileNotFoundError when called
    mock_joblib_load.side_effect = FileNotFoundError("Mocked file not found")

    # The app is designed to exit() if the model fails to load.
    # We can check if the test runner catches the SystemExit exception.
    with pytest.raises(SystemExit):
        # We need to re-import the app module to trigger the model loading logic again
        from importlib import reload
        from app import Iris_predictor
        reload(Iris_predictor)
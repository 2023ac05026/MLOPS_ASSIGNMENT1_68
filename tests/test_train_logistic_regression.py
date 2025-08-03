import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression

# Import the function to be tested
from src.train_logistic_regression import train_model

@pytest.fixture
def mock_iris_data(tmpdir):
    """Creates a temporary CSV file with mock iris data for testing."""
    # Create a temporary directory and file path
    data_dir = tmpdir.mkdir("data")
    file_path = data_dir.join("processed_iris.csv")
    
    # Create a simple DataFrame that resembles the processed iris data
    mock_df = pd.DataFrame({
        'sepal length (cm)': np.random.rand(50) * 2 + 4,
        'sepal width (cm)': np.random.rand(50) * 2 + 2,
        'petal length (cm)': np.random.rand(50) * 5 + 1,
        'petal width (cm)': np.random.rand(50) * 2,
        'target': np.random.randint(0, 3, 50)
    })
    mock_df.to_csv(file_path, index=False)
    return str(file_path)

def test_data_loading_and_splitting(mock_iris_data):
    """
    Test Case 1: Ensures the function can load data and returns a trained model.
    This is a basic "smoke test" to see if the function runs without errors.
    """
    # We don't care about the output here, just that it runs
    model, accuracy = train_model(mock_iris_data)
    
    # Assert that the function returns a model and a float for accuracy
    assert model is not None
    assert isinstance(accuracy, float)

def test_model_is_logistic_regression(mock_iris_data):
    """
    Test Case 2: Verifies that the best model found by GridSearchCV is a
    LogisticRegression instance.
    """
    model, _ = train_model(mock_iris_data)
    
    assert isinstance(model, LogisticRegression)

def test_model_is_trained(mock_iris_data):
    """
    Test Case 3: Checks if the returned model has been fitted by accessing
    an attribute that only exists after training (like .coef_).
    """
    model, _ = train_model(mock_iris_data)
    
    # The 'coef_' attribute is created when the model is fitted.
    # Accessing it will fail if the model is not trained.
    assert hasattr(model, 'coef_')

def test_accuracy_is_within_valid_range(mock_iris_data):
    """
    Test Case 4: Confirms that the returned accuracy score is a valid
    probability (between 0.0 and 1.0).
    """
    _, accuracy = train_model(mock_iris_data)
    
    assert 0.0 <= accuracy <= 1.0

@patch('src.train_logistic_regression.mlflow')
def test_mlflow_logging_is_called(mock_mlflow, mock_iris_data):
    """
    Test Case 5: Mocks the MLflow library to verify that the key logging
    functions (autolog, start_run, log_params, log_metric) are called.
    """
    # Run the training function
    train_model(mock_iris_data)

    # Assert that the main MLflow functions were called at least once
    mock_mlflow.autolog.assert_called_once()
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_params.assert_called_once()
    mock_mlflow.log_metric.assert_called_with("test_set_accuracy", pytest.approx(0.0, abs=1.0))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.ensemble import RandomForestClassifier

# Import the function to be tested
from src.train_random_forest import train_rf_model

@pytest.fixture
def mock_iris_data(tmpdir):
    """Creates a temporary CSV file with mock iris data for testing."""
    data_dir = tmpdir.mkdir("data")
    file_path = data_dir.join("processed_iris.csv")
    
    mock_df = pd.DataFrame({
        'sepal length (cm)': np.random.rand(50) * 2 + 4,
        'sepal width (cm)': np.random.rand(50) * 2 + 2,
        'petal length (cm)': np.random.rand(50) * 5 + 1,
        'petal width (cm)': np.random.rand(50) * 2,
        'target': np.random.randint(0, 3, 50)
    })
    mock_df.to_csv(file_path, index=False)
    return str(file_path)

def test_function_runs_without_error(mock_iris_data):
    """
    Test Case 1: A basic "smoke test" to ensure the function runs and returns
    the expected types (a model and a float).
    """
    model, accuracy = train_rf_model(mock_iris_data)
    
    assert model is not None
    assert isinstance(accuracy, float)

def test_model_is_random_forest(mock_iris_data):
    """
    Test Case 2: Verifies that the best model found by GridSearchCV is a
    RandomForestClassifier instance.
    """
    model, _ = train_rf_model(mock_iris_data)
    
    assert isinstance(model, RandomForestClassifier)

def test_model_is_trained(mock_iris_data):
    """
    Test Case 3: Checks if the returned model has been fitted by accessing
    an attribute that only exists after training (like .estimators_).
    """
    model, _ = train_rf_model(mock_iris_data)
    
    # The 'estimators_' attribute is a list of the trees in the forest
    # and is created when the model is fitted.
    assert hasattr(model, 'estimators_')
    assert len(model.estimators_) > 0

def test_accuracy_is_within_valid_range(mock_iris_data):
    """
    Test Case 4: Confirms that the returned accuracy score is a valid
    probability (between 0.0 and 1.0).
    """
    _, accuracy = train_rf_model(mock_iris_data)
    
    assert 0.0 <= accuracy <= 1.0

@patch('src.train_random_forest.mlflow')
def test_mlflow_logging_is_called(mock_mlflow, mock_iris_data):
    """
    Test Case 5: Mocks the MLflow library to verify that the key logging
    functions are called during the training process.
    """
    train_rf_model(mock_iris_data)

    # Assert that the main MLflow functions were called at least once
    mock_mlflow.autolog.assert_called_once()
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_params.assert_called_once()
    mock_mlflow.log_metric.assert_called_with("test_set_accuracy", pytest.approx(0.0, abs=1.0))
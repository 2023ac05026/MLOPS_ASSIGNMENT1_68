import pytest
import pandas as pd
import os
import numpy as np
from unittest.mock import patch

# Add src directory to path to import the script
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pre_processing import preprocess_data

@pytest.fixture
def temp_paths(tmpdir):
    """A pytest fixture to create temporary file paths for testing."""
    raw_path = tmpdir.join("raw_iris.csv")
    processed_path = tmpdir.join("processed_iris.csv")
    return str(raw_path), str(processed_path)

@patch('src.pre_processing.load_iris')
def test_file_creation(mock_load_iris, temp_paths):
    """
    Test Case 1: Verifies that both the raw and processed CSV files are created.
    """
    raw_path, processed_path = temp_paths
    
    # Setup mock data to be returned by load_iris
    mock_iris = {
        'data': np.array([[5.1, 3.5, 1.4, 0.2]]),
        'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
        'target': np.array([0]),
        'target_names': np.array(['setosa'])
    }
    mock_load_iris.return_value = pd.Series(mock_iris)

    preprocess_data(raw_path, processed_path)

    assert os.path.exists(raw_path), "Raw data file should be created"
    assert os.path.exists(processed_path), "Processed data file should be created"

@patch('src.pre_processing.load_iris')
def test_missing_value_handling(mock_load_iris, temp_paths):
    """
    Test Case 2: Checks if missing numerical values (NaN) are correctly filled.
    """
    raw_path, processed_path = temp_paths
    
    mock_data = np.array([[5.1, 3.5, 1.4, 0.2], [np.nan, 3.0, 1.5, 0.3]])
    mock_iris = {
        'data': mock_data,
        'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
        'target': np.array([0, 0]),
        'target_names': np.array(['setosa'])
    }
    mock_load_iris.return_value = pd.Series(mock_iris)

    processed_df = preprocess_data(raw_path, processed_path)
    
    # Check that no NaN values exist in the final numeric columns
    numeric_cols = processed_df.select_dtypes(include=np.number).columns
    assert not processed_df[numeric_cols].isnull().values.any(), "Missing values should be filled"

@patch('src.pre_processing.load_iris')
def test_target_encoding(mock_load_iris, temp_paths):
    """
    Test Case 3: Ensures the 'species' column is correctly converted to a numerical 'target' column.
    """
    raw_path, processed_path = temp_paths
    
    mock_iris = {
        'data': np.array([[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5]]),
        'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
        'target': np.array([0, 1, 2]),
        'target_names': np.array(['setosa', 'versicolor', 'virginica'])
    }
    mock_load_iris.return_value = pd.Series(mock_iris)
    
    processed_df = preprocess_data(raw_path, processed_path)
    
    assert 'target' in processed_df.columns, "Processed data should have a 'target' column"
    assert 'species' not in processed_df.columns, "Original 'species' column should be removed or replaced"
    assert sorted(processed_df['target'].unique()) == [0, 1, 2], "Target column should be encoded as 0, 1, 2"

@patch('src.pre_processing.load_iris')
def test_feature_scaling(mock_load_iris, temp_paths):
    """
    Test Case 4: Verifies that numerical features are standardized (mean close to 0, std dev close to 1).
    """
    raw_path, processed_path = temp_paths

    from sklearn.datasets import load_iris as actual_load_iris
    iris_data = actual_load_iris()
    mock_load_iris.return_value = iris_data

    processed_df = preprocess_data(raw_path, processed_path)
    
    feature_columns = iris_data.feature_names
    
    for col in feature_columns:
        assert np.isclose(processed_df[col].mean(), 0, atol=1e-9), f"Mean of scaled feature '{col}' should be close to 0"
        assert np.isclose(processed_df[col].std(ddof=0), 1, atol=1e-9), f"Std dev of scaled feature '{col}' should be close to 1"

def test_no_data_leakage(temp_paths):
    """
    Test Case 5: Ensures the raw and processed dataframes have the same number of rows.
    """
    raw_path, processed_path = temp_paths
    
    preprocess_data(raw_path, processed_path)
    
    df_raw = pd.read_csv(raw_path)
    df_processed = pd.read_csv(processed_path)
    
    assert len(df_raw) == len(df_processed), "Processed data should have the same number of rows as raw data"

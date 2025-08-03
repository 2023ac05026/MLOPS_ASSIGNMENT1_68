#!/usr/bin/env python
from sklearn.datasets import load_iris
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def preprocess_data(raw_data_path, processed_data_path):
    """
    Loads Iris data, saves a raw version, processes it, and saves the processed version.

    Processing includes:
    - Creating a human-readable target column.
    - Handling missing values by filling with the mean.
    - Encoding the categorical target variable.
    - Scaling features using StandardScaler.

    Args:
        raw_data_path (str): Path to save the raw CSV file.
        processed_data_path (str): Path to save the processed CSV file.
    
    Returns:
        pd.DataFrame: The processed and scaled DataFrame.
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    # Load dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')

    # Map numerical labels to species names for the raw file
    target_names_map = {i: name for i, name in enumerate(iris.target_names)}
    y_named = y.map(target_names_map)

    # Combine for the raw DataFrame
    df_raw = X.copy()
    df_raw['species'] = y_named
    
    # Save raw CSV
    df_raw.to_csv(raw_data_path, index=False)
    print(f"Iris dataset saved to {raw_data_path}")

    # --- Start Processing ---
    df_to_process = df_raw.copy()

    # Step 3: Check for missing values
    if df_to_process.isnull().values.any():
        print("Missing values found. Filling with column mean...")
        numeric_cols = df_to_process.select_dtypes(include=np.number).columns
        df_to_process[numeric_cols] = df_to_process[numeric_cols].fillna(df_to_process[numeric_cols].mean())
    else:
        print("No missing values found.")

    # Step 4: Encode categorical target
    label_encoder = LabelEncoder()
    df_to_process['target'] = label_encoder.fit_transform(df_to_process['species'])
    
    # Step 5: Feature Scaling (Standardization)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_to_process[iris.feature_names])
    df_scaled = pd.DataFrame(features_scaled, columns=iris.feature_names)
    df_scaled['target'] = df_to_process['target']

    df_scaled.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")
    
    return df_scaled

if __name__ == '__main__':
    raw_path = os.path.join("Data", "raw", "iris.csv")
    processed_path = os.path.join("Data", "processed", "processed_iris.csv")
    preprocess_data(raw_data_path=raw_path, processed_data_path=processed_path)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from frouros.detectors.concept_drift import DDM, DDMConfig
import pandas as pd
import logging
import time

def detect_and_visualize_drift(X, y, split_idx=10000, warning_level=2.0, drift_level=3.0, min_num_instances=2000):
    """
    Detects concept drift using the DDM algorithm and visualizes results.

    Args:
        X (numpy.ndarray): Features of the dataset.
        y (numpy.ndarray): Labels of the dataset.
        split_idx (int): Number of samples to use as reference data before drift detection.
        warning_level (float): Warning threshold for DDM.
        drift_level (float): Drift threshold for DDM.
        min_num_instances (int): Minimum samples required before drift detection.

    Returns:
        dict: A dictionary containing drift points, warning points, and prediction errors.
    """
    # Ensure that all columns in X are numeric and y is a string
    X = np.array(X, dtype=np.float64)  # Ensure X is of type float64
    y = np.array(y, dtype=str)  # Ensure y is of type string
    
    print("Feature data type:", X.dtype)
    print("Label data type:", y.dtype)

    # Clean the data: Replace NaN values with 0 for simplicity
    if np.isnan(X).any() or np.isnan(y).any():
        print("Data contains NaN values. Cleaning the dataset...")
        X = np.nan_to_num(X)  # Replace NaNs with 0
        y = np.nan_to_num(y)

    # Split dataset: Use the first part as reference
    X_ref, y_ref, X_test, y_test = (
        X[:split_idx],
        y[:split_idx].ravel(),
        X[split_idx:],
        y[split_idx:].ravel(),
    )

    # Build a classification pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])
    pipeline.fit(X=X_ref, y=y_ref)

    # Configure the DDM drift detector
    config = DDMConfig(
        warning_level=warning_level,
        drift_level=drift_level,
        min_num_instances=min_num_instances
    )

    # Initialize variables
    drift_points = []
    warning_points = []
    errors = []

    # Apply drift detection
    start_index = 0
    while start_index < len(X_test):
        # Initialize detector for the current segment
        detector = DDM(config=config)

        for i, (X_sample, y_sample) in enumerate(zip(X_test[start_index:], y_test[start_index:]), start=start_index):
            y_pred = pipeline.predict(X_sample.reshape(1, -1))  # Predict on the sample
            error = 1 - (y_pred.item() == y_sample)  # Calculate prediction error
            detector.update(value=error)  # Update the drift detector with the error value

            # Append error for plotting
            errors.append(error)

            # Check for drift or warning
            status = detector.status
            if status["drift"]:  # Drift detected
                drift_points.append(i)  # Store drift point index
                print(f"Drift detected at index {i}")

                # Reset starting point for next drift detection
                start_index = i + 1
                break  # Exit loop to reset the detector

            elif status["warning"]:  # Warning threshold exceeded
                warning_points.append(i)  # Store warning point index
        else:
            # If no drift is detected in the remaining data, break the loop
            break

    # Visualization (optional, can be implemented as needed)
    # Plot errors, drift points, and warning points for analysis.

    # Return results
    print("Drift points:", drift_points)
    return {
        "drift_points": drift_points,
        "warning_points": warning_points,
        "errors": errors
    }
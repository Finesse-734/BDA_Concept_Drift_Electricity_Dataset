
import numpy as np
import logging
from frouros.detectors.concept_drift import DDM, DDMConfig

class ConceptDriftDetector:
    def __init__(self, pipeline, warning_level=2.0, drift_level=3.0, split_idx=10000):
        """
        Initialize the drift detection mechanism with DDM and a classification pipeline.

        Args:
            pipeline: Pre-trained classification pipeline (e.g., Logistic Regression with preprocessing).
            warning_level (float): Warning threshold for DDM.
            drift_level (float): Drift threshold for DDM.
            split_idx (int): Number of instances to use as reference data before drift detection.
        """
        self.pipeline = pipeline
        self.split_idx = split_idx
        self.config = DDMConfig(
            warning_level=warning_level,
            drift_level=drift_level,
            min_num_instances=2000  # Minimum samples required before detecting drift
        )
        self.detector = DDM(config=self.config)
        self.drift_points = []  # To store all drift indices
        self.warning_points = []  # To store all warning indices

    def check_drift(self, X, y):
        """
        Detects concept drift in a batch of data.

        Args:
            X (numpy.ndarray): Features of the batch.
            y (numpy.ndarray): Labels of the batch.

        Returns:
            dict: Contains drift points, warning points, and prediction errors.
        """
        # Ensure X and y are numpy arrays of the correct type
        X = np.array(X)
        y = np.array(y).astype(str)  # Ensure y is in string format for classification tasks

        # Split dataset into reference and test parts
        X_ref, y_ref, X_test, y_test = (
            X[:self.split_idx],
            y[:self.split_idx].ravel(),
            X[self.split_idx:],
            y[self.split_idx:].ravel(),
        )

        # Fit the pipeline on the reference data
        self.pipeline.fit(X=X_ref, y=y_ref)

        # Initialize result storage
        drift_points = []
        warning_points = []
        errors = []

        # Start drift detection on test data
        for i, (X_sample, y_sample) in enumerate(zip(X_test, y_test)):
            # Predict on the current sample
            y_pred = self.pipeline.predict(X_sample.reshape(1, -1))
            error = 1 - (y_pred.item() == y_sample)  # Calculate prediction error
            self.detector.update(value=error)  # Update the drift detector with the error value

            # Append the error for visualization
            errors.append(error)

            # Check for drift or warning
            status = self.detector.status
            if status["drift"]:
                drift_points.append(i)  # Store drift point index
                logging.warning(f"Drift detected at index {i}.")
                print("********************************************\n")
                print("********************************************\n")
                print("********************************************\n")
                print("********************************************\n")
                print("********************************************\n")
                print("********************************************\n")
                print("********************************************\n")
                print("********************************************\n")
                print("********************************************\n")
                self.detector.reset()  # Reset the detector after drift detection
                break  # Exit loop to reset the detector

            elif status["warning"]:
                warning_points.append(i)  # Store warning point index

        return {"drift_points": drift_points, "warning_points": warning_points, "errors": errors}

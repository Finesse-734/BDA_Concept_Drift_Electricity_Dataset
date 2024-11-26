import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from frouros.detectors.concept_drift import DDM, DDMConfig

# Load dataset
data_file = "elec2_data.dat"
label_file = "elec2_label.dat"

# Read data files
data_columns = ["day", "period", "nswdemand", "vicprice", "vicdemand", "transfer"]
data_info = pd.read_csv(data_file, delim_whitespace=True, names=data_columns, header=None)
label_info = pd.read_csv(label_file, names=["class"], header=None)

# Combine data and labels
data_info.reset_index(drop=True, inplace=True)
label_info.reset_index(drop=True, inplace=True)
dataset = pd.concat([data_info, label_info], axis=1)

# Extract relevant features and labels
X = dataset[["nswdemand", "vicprice", "transfer"]].values
y = dataset["class"].astype(str).values  # Convert class column to string

# Split dataset: Use the first 20,000 samples as reference
split_idx = 10000
X_ref, y_ref, X_test, y_test = (
    X[:split_idx],
    y[:split_idx].ravel(),
    X[split_idx:],
    y[split_idx:].ravel(),
)

# Build a classification pipeline
pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
pipeline.fit(X=X_ref, y=y_ref)

# Configure the DDM drift detector
config = DDMConfig(
    warning_level=2.0,  # Warning threshold
    drift_level=3.0,    # Drift threshold
    min_num_instances=2000,  # Minimum samples required before detecting drift
)

# Initialize variables
drift_points = []  # To store all drift indices
errors = []  # Store prediction errors for visualization
warning_points = []  # Indices where warnings occur

# Apply drift detection
start_index = 0  # Starting point for drift detection
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

# Plot the error rate and all drift points
plt.figure(figsize=(12, 6))
plt.plot(errors, label="Prediction Error", color="blue", alpha=0.7)
plt.scatter(drift_points, [errors[i - split_idx] for i in drift_points], color="red", label="Drift Detected", zorder=5)
plt.scatter(warning_points, [errors[i - split_idx] for i in warning_points], color="orange", label="Warning Level", zorder=5)
plt.axhline(y=0.5, color="gray", linestyle="--", label="Error Threshold (0.5)")
plt.title("Prediction Error and Concept Drift Detection")
plt.xlabel("Index")
plt.ylabel("Error")
plt.legend()
plt.show()

# Print all drift points
if drift_points:
    print("All detected drift points:", drift_points)
else:
    print("No drift detected in the dataset.")
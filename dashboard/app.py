import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# File path to drift results
DRIFT_RESULTS_FILE = "dashboard/drift_results.csv"

# Streamlit app title
st.title("Concept Drift Detection Dashboard")
st.write("This dashboard visualizes concept drift detected in streaming data.")

# Load the drift results file dynamically
def load_drift_results():
    if os.path.exists(DRIFT_RESULTS_FILE):
        try:
            # Attempt to load the file
            drift_results = pd.read_csv(DRIFT_RESULTS_FILE, names=["batch_id", "drift_index"])
            if drift_results.empty:
                st.warning("The drift results file exists but contains no data.")
                return pd.DataFrame(columns=["batch_id", "drift_index"])
            return drift_results
        except Exception as e:
            st.error(f"Error reading drift results file: {e}")
            return pd.DataFrame(columns=["batch_id", "drift_index"])
    else:
        st.warning(f"No drift results file found at {DRIFT_RESULTS_FILE}.")
        return pd.DataFrame(columns=["batch_id", "drift_index"])

# Main app logic
st.sidebar.header("Dashboard Settings")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 5)

st.subheader("Drift Results")
placeholder = st.empty()

while True:
    # Load drift results
    drift_results = load_drift_results()

    if not drift_results.empty:
        with placeholder.container():
            # Display drift results in a table
            st.write("### Detected Drift Points")
            st.dataframe(drift_results)

            # Visualization: Scatter plot for drift points
            st.write("### Drift Points Visualization")
            fig, ax = plt.subplots()
            ax.scatter(drift_results["drift_index"], drift_results["batch_id"], c="red", label="Drift Points")
            ax.set_xlabel("Drift Index")
            ax.set_ylabel("Batch ID")
            ax.set_title("Drift Points Detected")
            ax.legend()
            st.pyplot(fig)
    else:
        with placeholder.container():
            st.write("No drift points detected yet. Waiting for updates...")

    st.write(f"Last updated: {pd.Timestamp.now()}")

    # Pause for the refresh rate
    time.sleep(refresh_rate)

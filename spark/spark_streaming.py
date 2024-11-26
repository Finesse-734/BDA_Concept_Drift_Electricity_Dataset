from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import os
import sys
import pandas as pd
import logging
import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Dynamically add the project root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from ml_models.drift_detector import detect_and_visualize_drift  # Import the function

# File paths
DRIFT_RESULTS_FILE = "dashboard/drift_results.csv"  # File to log drift detection results
DRIFT_LOG_FILE = "logs/drift_log.txt"              # Log file for drift detection events

# Define schema for Kafka data
schema = StructType([
    StructField("day", StringType(), True),
    StructField("period", FloatType(), True),
    StructField("nswdemand", FloatType(), True),
    StructField("vicprice", FloatType(), True),
    StructField("vicdemand", FloatType(), True),
    StructField("transfer", FloatType(), True),
    StructField("label", StringType(), True)
])

# Initialize Spark session
spark = SparkSession.builder \
    .appName("KafkaPySparkStreaming") \
    .getOrCreate()

# Kafka stream configuration
kafka_stream = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "electricity_topic1") \
    .option("startingOffsets", "earliest") \
    .option("maxOffsetsPerTrigger", 20000) \
    .load()

# Parse JSON messages from Kafka
parsed_stream = kafka_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Create the pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])

# Placeholder for cumulative batch data
cumulative_data = pd.DataFrame(columns=["nswdemand", "vicprice", "transfer", "label"])

# Function to process each batch
def process_batch(batch_df, batch_id):
    global cumulative_data  # Access the global variable for cumulative storage
    try:
        # Convert Spark DataFrame to Pandas DataFrame
        pd_data = batch_df.toPandas()

        # Append the batch to cumulative data
        if not pd_data.empty:
            cumulative_data = pd.concat([cumulative_data, pd_data], ignore_index=True)

            # If cumulative data reaches 10,000 records, process drift detection
            if len(cumulative_data) >= 10000:
                # Extract features and labels
                X = cumulative_data[["nswdemand", "vicprice", "transfer"]].values
                y = cumulative_data["label"].astype(str).values  # Ensure labels are strings

                # Call the drift detection function
                results = detect_and_visualize_drift(X, y, split_idx=10000)  # Adjust split_idx if needed

                # Log drift points
                drift_points = results["drift_points"]
                with open(DRIFT_LOG_FILE, "a") as log_file:
                    log_file.write(
                        f"Batch {batch_id} - Drift Points: {drift_points} at {time.ctime()}\n"
                    )

                # Save only drift points to the drift results file
                if drift_points:
                    drift_results_df = pd.DataFrame({
                        "batch_id": [batch_id] * len(drift_points),
                        "drift_index": drift_points
                    })
                    drift_results_df.to_csv(DRIFT_RESULTS_FILE, mode="a", header=False, index=False)

                # Clear cumulative data after processing
                cumulative_data = pd.DataFrame(columns=["nswdemand", "vicprice", "transfer", "label"])
        else:
            logging.info(f"Batch {batch_id} is empty.")

    except Exception as e:
        logging.error(f"Error processing batch {batch_id}: {str(e)}")

    # Simulate real-time streaming
    time.sleep(1)

# Apply the process_batch function to each micro-batch
query = parsed_stream.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .trigger(processingTime='10 seconds') \
    .start()

# Await termination to keep the stream running
query.awaitTermination()

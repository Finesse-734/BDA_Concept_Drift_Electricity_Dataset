from confluent_kafka import Producer
import json
import pandas as pd
import time

# Initialize Kafka Producer
producer = Producer({'bootstrap.servers': 'localhost:9092'})

# Load the dataset
data_path = '../data/elec2_data.dat'  
labels_path = '../data/elec2_label.dat' 

data_columns = ['day', 'period', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']
data = pd.read_csv(data_path, sep=',', names=data_columns, header=None)

labels = pd.read_csv(labels_path, header=None, names=['label'])

if not labels.empty:
    data['label'] = labels['label']

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def send_data_to_kafka():
    for index, row in data.iterrows():
        data_point = {
            "day": row['day'],
            "period": row['period'],
            "nswdemand": row['nswdemand'],
            "vicprice": row['vicprice'],
            "vicdemand": row['vicdemand'],
            "transfer": row['transfer'],
        }
        
        if 'label' in row:
            data_point["label"] = row['label']

        print(f"Sending: {data_point}")
        
        producer.produce('electricity_topic1', value=json.dumps(data_point).encode('utf-8'), callback=delivery_report)

        producer.flush()

if __name__ == "__main__":
    send_data_to_kafka()

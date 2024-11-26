from confluent_kafka import Consumer, KafkaException
import json

# Configure the Kafka Consumer
consumer_conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'electricity_group',  # Ensure it matches the producer's group ID if needed
    'auto.offset.reset': 'earliest',  # Start reading from the beginning if no offset is available
    'enable.auto.commit': True  # Commit offsets automatically
}

# Initialize the Kafka Consumer
consumer = Consumer(consumer_conf)

# Subscribe to the topic
consumer.subscribe(['electricity_topic1'])  # Match the topic name with producer

print("Consumer is listening for messages on 'electricity_topic1'...")

# Poll messages from Kafka and display them
try:
    while True:
        msg = consumer.poll(timeout=5.0) 
        if msg is None:
            print("hello")
            continue 

        if msg.error():
            if msg.error().code() == KafkaException._PARTITION_EOF:
                print(f"End of partition reached {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
            else:
                raise KafkaException(msg.error())  # Raise other errors

        data = json.loads(msg.value().decode('utf-8'))
        print(f"Received: {data}")

except KeyboardInterrupt:
    print("Consumer interrupted by user")

finally:
    consumer.close()

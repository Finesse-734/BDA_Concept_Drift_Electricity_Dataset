from confluent_kafka.admin import AdminClient, NewTopic, KafkaException

admin_client = AdminClient({"bootstrap.servers": "localhost:9092"})

new_topic = NewTopic(
    topic="electricity_topic1", 
    num_partitions=1, 
    replication_factor=1
)

try:
    admin_client.create_topics([new_topic])
    print(f"Topic '{new_topic.topic}' created successfully.")
except KafkaException as e:
    print(f"Failed to create topic '{new_topic.topic}': {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

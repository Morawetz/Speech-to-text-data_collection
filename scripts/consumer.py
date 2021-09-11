from time import sleep
from json import dumps
from kafka import KafkaConsumer
from json import loads

consumer = KafkaConsumer('group6_test',
client_id='d_id',
bootstrap_servers=['b-1.demo-cluster-1.9q7lp7.c1.kafka.eu-west-1.amazonaws.com:9092'],
auto_offset_reset='earliest',
enable_auto_commit=False)


for event in consumer:
    event_data = event.value
    # Do whatever you want
    print(event_data)
    sleep(2)

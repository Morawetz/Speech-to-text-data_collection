from time import sleep
from kafka import KafkaConsumer
import numpy as np
import io
from pydub import AudioSegment
import boto3
s3 = boto3.resource('s3')

consumer = KafkaConsumer('morawetz_audio_topic',
client_id='d_id',
bootstrap_servers=['b-1.demo-cluster-1.9q7lp7.c1.kafka.eu-west-1.amazonaws.com:9092','b-2.demo-cluster-1.9q7lp7.c1.kafka.eu-west-1.amazonaws.com:9092'],
auto_offset_reset='earliest',
enable_auto_commit=False)


for event in consumer:
    event_data = event.value
    bytes_wav = bytes()
    byte_io = io.BytesIO(event_data)
    print("completed")
    audio = AudioSegment.from_raw(byte_io, sample_width=2, frame_rate=22050, channels=1).export("test", format='wav')
    s3.meta.client.upload_file('test', 'fumbabucket', 'test.wav')
    sleep(2)

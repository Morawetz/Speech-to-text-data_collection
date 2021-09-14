from datetime import timedelta
from textwrap import dedent
from kafka import KafkaConsumer
from time import sleep
import numpy as np
import io
#from airflow.providers.docker.operators.docker import DockerOperator
#from pydub import AudioSegment
import boto3
s3 = boto3.resource('s3')
# instantiate a DAG
from airflow import DAG

# Operators
#from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}

def kafka_to_s3():
    consumer = KafkaConsumer('morawetz_audio_topic',
    client_id='d_id',
    bootstrap_servers=['b-1.demo-cluster-1.9q7lp7.c1.kafka.eu-west-1.amazonaws.com:9092','b-2.demo-cluster-1.9q7lp7.c1.kafka.eu-west-1.amazonaws.com:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=False)
    
    for event in consumer:
        s3.meta.client.upload_file('/usr/local/airflow/dags/test.wav', 'fumbabucket', 'test1.wav')
        print("Saved in bucket successfully!")

with DAG(
    'audio',
    default_args=default_args,
    description='A DAG script that schedules audio',
    schedule_interval=timedelta(minutes=10),
    start_date=days_ago(1),
    tags=['audio'],
) as dag:
    
    src_s3 = PythonOperator(
        task_id='kafka_to_s3', 
        python_callable=kafka_to_s3, 
        dag=dag)
    
   
    
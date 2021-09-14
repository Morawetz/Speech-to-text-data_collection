## Speech-to-text-data_collection

## Introduction
In this project,we are going to design and build a robust, large scale, fault tolerant, highly available Kafka cluster that can be used to
post a sentence and receive an audio file.

By the end of this project we will produce a tool that can be deployed to process posting and receiving text and audio files from and into a data lake, apply
transformation in a distributed manner, and load it into a warehouse in a suitable format to train a speech-to-text model.
![index](https://user-images.githubusercontent.com/47286297/133191700-346187df-e2c1-4a61-a5bd-60b81fe72dc8.png)


## Table of Contents
  - [Introduction](#Introduction)
  - [Technologies Used](#Technologies)
  - [ Volunteer](#Volunteer)
  - [Folders](#Folders)
  - [Architecture](#Architecture)

## Technologies
  - [Apache Kafka](https://kafka.apache.org/documentation/#quickstart_download): To sequentially log streaming data into specific topics 
  - [Apache Airflow](https://airflow.apache.org/docs/apache-airflow/stable/installation.html): To create,ocherstrate and monitor data workflows 
  - [Apache Spark](https://spark.apache.org/downloads.html):To transform and load  data from Kafka cluster
  - [S3 Buckets](): For storing transformed streaming data 

## Volunteer
To help us in collecting data audio data for Amharic language, visit [datacollectionpipeline](https://datacollectionpipeline.herokuapp.com/).
On the home page, go to 'CONTRIBUTE AUDIO'.

You will be presented with a statement in Amharic. Click on 'Record' and read the statement out loud. Once you have finished, click 'Stop' and send. 
![Record Aduio](https://github.com/Morawetz/Speech-to-text-data_collection/blob/main/screenshots/stopped.png)

## Architecture
Following is a detailed technical diagram showing the configuration of the archictecure.
![Architecture](https://github.com/Morawetz/Speech-to-text-data_collection/blob/documentation/screenshots/data_pipelinne.png)
  

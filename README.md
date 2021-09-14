## Speech-to-text-data_collection

## Introduction
In this project,we are going to design and build a robust, large scale, fault tolerant, highly available Kafka cluster that can be used to
post a sentence and receive an audio file.

By the end of this project we will produce a tool that can be deployed to process posting and receiving text and audio files from and into a data lake, apply
transformation in a distributed manner, and load it into a warehouse in a suitable format to train a speech-to-text model.

## Table of Contents
  - [Introduction](#Introduction)
  - [Technologies Used](#Technologies)
  - [Installation](#Installation)

## Technologies
  - [Apache Kafka](): To sequentially log streaming data into specific topics 
  - [Apache Airflow](): To create,ocherstrate and monitor data workflows 
  - [S3 Buckets](): For storing transformed streaming data 
  - [Airflow](): To programmatically author, schedule and monitor workflows.


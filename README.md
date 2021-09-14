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
  - [Installation](#Installation)
  - [Folders](#Folders)

## Technologies
  - [Apache Kafka](): To sequentially log streaming data into specific topics 
  - [Apache Airflow](): To create,ocherstrate and monitor data workflows 
  - [Apache Spark]():To transform and load from Kafka cluster
  - [S3 Buckets](): For storing transformed streaming data 

## Installation
  <a href="https://kafka.apache.org/documentation/#quickstart_download" target="_blank">[Apache Kafka]:</a>
  <a href="https://airflow.apache.org/docs/apache-airflow/stable/installation.html">[Apache Airflow]:</a> 
  

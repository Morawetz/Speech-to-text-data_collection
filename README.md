## Data Collection

## Introduction
In this project,we have designed and built a robust, large scale, fault tolerant, highly available data pipeline that is used to collect audio data given a text transcript.

By the end of this project we will produce a tool that can be deployed to process posting and receiving text and audio files from and into a data lake, apply
transformation in a distributed manner, and load it into a warehouse in a suitable format to train a speech-to-text model.

## Table of Contents
  - [Introduction](#Introduction)
  - [Technologies Used](#Technologies)
  - [Quick Tour](#QuickTour)
  - [Volunteer](#Volunteer)
  - [Architecture](#Architecture)

## Technologies
  - [Apache Kafka](): To sequentially log streaming data into specific topics 
  - [Apache Airflow](): To create,ocherstrate and monitor data workflows 
  - [Apache Spark](): For distributed processing system.
  - [S3 Buckets](): For storing transformed streaming data 
  - [Airflow](): To programmatically author, schedule and monitor workflows.

## Quick Tour 

## Volunteer
To help us in collecting data audio data for Amharic language, visit [datacollectionpipeline](https://datacollectionpipeline.herokuapp.com/).
On the home page, go to 'CONTRIBUTE AUDIO'.

You will be presented with a statement in Amharic. Click on 'Record' and read the statement out loud. Once you have finished, click 'Stop' and send. 
![Record Aduio](https://github.com/Morawetz/Speech-to-text-data_collection/blob/main/screenshots/stopped.png)

## Architecture
Following is a detailed technical diagram showing the configuration of the archictecure.
![Architecture](https://github.com/Morawetz/Speech-to-text-data_collection/blob/documentation/screenshots/data_pipelinee.png)

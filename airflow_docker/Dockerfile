FROM puckel/docker-airflow

USER root
RUN apt-get update

RUN apt install libasound2-dev python-dev python-numpy python-setuptools libsndfile-dev -y sudo
RUN pip install librosa
WORKDIR /usr/local/airflow/
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install jiwer
# 
# RUN apt-get update  \
#  && apt-get install -y sudo


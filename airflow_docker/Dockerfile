FROM puckel/docker-airflow

USER root
RUN apt-get update

RUN apt install libasound2-dev python-dev python-numpy python-setuptools libsndfile-dev -y sudo
RUN pip install librosa
WORKDIR /usr/local/airflow/
COPY requirements.txt .
RUN pip --default-timeout=100 install -r requirements.txt
# pip --default-timeout=100 install

# install cert because tf couldnt install
RUN pip install pip_system_certs

RUN pip --default-timeout=10000 install tensorflow==2.3.0
RUN pip install jiwer
RUN pip install h5py==2.10.0
# 
# RUN apt-get update  \
#  && apt-get install -y sudo


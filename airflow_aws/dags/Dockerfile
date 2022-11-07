FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04
# # #WORKDIR ./cuda/ 
# USER root

# RUN sudo chmod 777 /home/ubuntu/airflow//dags
# RUN sudo chmod 777 /home/ubuntu/airflow//logs
# RUN sudo chmod 777 /home/ubuntu/airflow//plugins
# RUN sudo chmod 777 /var/run/docker.sock

# RUN sudo groupadd docker
# RUN sudo usermod -aG docker $USER
# RUN newgrp docker

# RUN apt-get update \
#   && sudo apt-get install -y apt-utils \
#   && apt-get install -y --no-install-recommends \
#          vim \
#   && apt-get autoremove -yqq --purge \
#   && apt-get clean \
#   && rm -rf /var/lib/apt/lists/* 
  # && apt-get install psmisc

USER $(id -u)
#COPY ./cuda/ ./cuda/

# RUN export AIRFLOW_DAEMON_USER="${AIRFLOW_DAEMON_USER:-airflow}"
# RUN export AIRFLOW_DAEMON_GROUP="${AIRFLOW_DAEMON_GROUP:-airflow}"


FROM apache/airflow:2.4.0-python3.8
# FROM apache/airflow:slim-2.4.0-python3.8 ##This doenst work 
RUN python -m pip install --upgrade pip
#WORKDIR ..
COPY requirements.txt .
# COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
#COPY ./cuda/ ./cuda/


# RUN apt-get update
# RUN apt install python3
# RUN apt-get -y install python3-pip
# WORKDIR /home/ubuntu/airflow//logs
# COPY /home/hojun/airflow/logs/requirements.txt .
# RUN pip install -r requirements.txt

 
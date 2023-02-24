FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04
# # #WORKDIR ./cuda/ 
# USER root


# RUN sudo groupadd docker
# RUN sudo usermod -aG docker $USER
# RUN newgrp docker

# RUN apt-get update \
#   && apt-get install -y --no-install-recommends \
#          vim \
#   && apt-get autoremove -yqq --purge \
#   && apt-get clean \
#   && rm -rf /var/lib/apt/lists/* \
#   && apt-get install psmisc


USER root
RUN sudo chmod -R 777 /home/airflow/.cache
RUN apt-get update
RUN apt-get -y install build-essential
RUN apt-get -y install g++ 
RUN apt-get -y install apt-utils
RUN apt-get -y install sudo
RUN apt-get -y install vim
RUN apt-get -y install lsof
RUN apt-get -y install psmisc
RUN apt-get -y install net-tools


#RUN sudo chmod 777 /opt/airflow/dags
#RUN sudo chmod 777 /opt/airflow/logs
#RUN sudo chmod 777 /opt/airflow/plugins
#RUN sudo chmod 777 /var/run/docker.sock
#"${AIRFLOW_UID:-50000}:${AIRFLOW_GID:-0}"
USER "${AIRFLOW_UID:-50000}:${AIRFLOW_GID:-0}"
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

#USER "${AIRFLOW_UID:-50000}:${AIRFLOW_GID:-0}" 

# RUN apt-get update
# RUN apt install python3
# RUN apt-get -y install python3-pip
# WORKDIR /opt/airflow/logs
# COPY /home/hojun/airflow/logs/requirements.txt .
# RUN pip install -r requirements.txt

 

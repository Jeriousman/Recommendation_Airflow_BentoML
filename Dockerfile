#FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04

#USER "${AIRFLOW_UID:-50000}:${AIRFLOW_GID:-0}"
##COPY ./cuda/ ./cuda/

#FROM apache/airflow:2.4.0-python3.8
#RUN python -m pip install --upgrade pip

#USER root
#RUN apt-get update
#RUN apt-get -y install dpkg
#RUN apt-get -y install build-essential
#RUN apt-get -y install libpq-dev
#RUN apt-get -y install g++-11
#RUN apt-get -y install apt-utils
#RUN apt-get -y install sudo
#RUN apt-get -y install vim
#RUN apt-get -y install lsof
#RUN apt-get -y install psmisc
#RUN apt-get -y install net-tools

#USER "${AIRFLOW_UID:-50000}:${AIRFLOW_GID:-0}" 
#COPY requirements.txt .
#RUN pip install -r requirements.txt

#USER root
#RUN sudo chmod -R 777 /home/airflow/.cache





#ARG USERNAME=hannah
#ARG AIRFLOW_UID=1000
#ARG AIRFLOW_GID=0





FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04

#RUN groupadd --gid $AIRFLOW_GID $USERNAME \
#    && useradd --uid $USER_UID --gid $AIRFLOW_GID -m $USERNAME



#USER airflow
FROM apache/airflow:2.5.3-python3.9
#FROM apache/airflow:2.4.0-python3.8 
#RUN python -m pip install --upgrade pip
COPY requirements.txt .
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         vim \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
#RUN sudo chmod -R 777 /home/airflow/.cache
RUN apt-get -y install apt-utils
RUN apt-get -y install sudo
RUN apt-get -y install vim
RUN apt-get -y install lsof
RUN apt-get -y install psmisc
RUN apt-get -y install net-tools
RUN apt-get -y install build-essential
RUN apt-get -y install libpq-dev
RUN apt-get -y install g++-11
USER airflow
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
#RUN pip install -r requirements.txt
USER root
RUN sudo chmod -R 777 /home/airflow/.cache
USER airflow

 

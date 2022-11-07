#!/bin/bash
docker-compose up airflow-init
airflow db init
docker-compose up -d

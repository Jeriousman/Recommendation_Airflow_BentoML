#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:00:42 2022

@author: hojun
"""

import sys
import os
import airflow
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
# from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from utils.slack_alert import SlackAlert
from functions import  data
from functions import  process_tensor_to_dataloader 
from functions import  calculate_emb_vectors
from functions import  make_bento_model
# from functions import  prepare_user_or_pik_data
from airflow.models.baseoperator import chain
from datetime import datetime, timedelta
import pandas as pd
from typing import List



slack = SlackAlert("#airflow-monitor", "xoxb-2654232235489-4145809392837-oy7FLFexlTrHAFbhHZ2ZDKIO")

default_args = {
    "owner": 'airflow',
    'email_on_failure' : False,
    'email_on_retry' : False,
    'email' : 'hojun.seo@pikurate.com',
    'retries': 0,
    'retry_delay' : timedelta(minutes=5),
    'on_success_callback': slack.success_msg,
    'on_failure_callback': slack.fail_msg


}

with DAG(
    dag_id='pik_rec_airflow',
    start_date=datetime(2022, 9 , 28),
    schedule_interval="0 * * * *", #"*/30 * * * *",  ##meaning every 2 hours. https://crontab.guru/ 매새벽3시에돌린다 (UTC타임)
    default_args=default_args,
    catchup=False,
    
) as dag:





    task_clear_bento = BashOperator(   ##이전 bentoml model들을 모두 지운다
            task_id="clear_bento",         ##
            bash_command=
            """ 
                bentoml delete pik_recommender_bento --yes; bentoml models delete pik_recommender_model --yes; echo "Job continued"  
            """
            )
    
    
    

    task_db_data_fetching = PythonOperator(
            task_id='db_data_fetching',
            python_callable=data.db_data_fetching,
            op_kwargs={
                'default_path' : '/opt/airflow/dags/data',
                'hostname' : 'prod-back.c5dkkbujxodg.ap-northeast-2.rds.amazonaws.com',
                'dbname' : 'pikurate',
                'username' : 'postgres',
                'password' : 'postgres',
                'portnumber' : 5432

            })
    
    
    
    ##raw data를 정제된 link_cat_pik으로 만들어준다
    task_data_process = PythonOperator(
        task_id="raw_data_preprocess",
        python_callable=data.raw_data_preprocess,
        op_kwargs={
            "path": '/opt/airflow/dags/data', ##실제 DB에서가져오는 raw data
        }
    )
    
    
    task_save_processed_data = PythonOperator(  ##없으나 마나한 태스크지만 에어플로우에서 제대로 저장되지 않는 이상현상이 발견 된 적이 있기 때문에 넣었다
        task_id="save_processed_data",
        python_callable=data.check_data,
        op_kwargs={
            "df": '/opt/airflow/dags/data/link_cat_pik.csv', ##실제 DB에서가져오는 raw data
            'processed_data_path': '/opt/airflow/dags/data/processed_data.csv'
        }
    )
    
    
    
 
    #raw data를 정제된 link_cat_pik으로 만들어준다
    '''
    링크타이틀을 huggingface로 embedding하여 데이터로더를 만든다 
    '''
    task_linktitle_data_process_to_torch = PythonOperator(
        task_id="linktitle_data_to_torch",
        python_callable=process_tensor_to_dataloader.process_sent_tensor_to_torchdata,
        op_kwargs={
            "tokenizer_name": 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking', 
            'processed_data': '/opt/airflow/dags/data/processed_data.csv',
            'tokenizing_col': 'link_title',
            'max_len' : 24,
            'return_tensors': 'pt',
            'padding': 'max_length',
            'truncation': True,
            'batch_size': 256,
            'saving_dataloader_path': '/opt/airflow/dags/data/link_title_dataloader.pickle'
        }
    )
    
    
    '''
    픽타이틀을 huggingface로 embedding하여 데이터로더를 만든다 
    '''
    task_piktitle_data_process_to_torch = PythonOperator(
        task_id="piktitle_data_to_torch",
        python_callable=process_tensor_to_dataloader.process_sent_tensor_to_torchdata,
        op_kwargs={
            "tokenizer_name": 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking', 
            'processed_data': '/opt/airflow/dags/data/processed_data.csv',
            'tokenizing_col': 'pik_title',
            'max_len' : 24,
            'return_tensors': 'pt',
            'padding': 'max_length',
            'truncation': True,
            'batch_size': 256,
            'saving_dataloader_path': '/opt/airflow/dags/data/pik_title_dataloader.pickle'
        }
    )
    
    
    task_calculate_linktitle_emb_vector_and_pik_vector = PythonOperator(
        task_id="calculate_linktitle_emb_vector",
        python_callable=calculate_emb_vectors.calculate_emb,
        op_kwargs={
            'default_path' : '/opt/airflow/dags',
            "processed_data_path": '/opt/airflow/dags/data/processed_data.csv', 
            'tokenizer_name': 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
            'model_name': 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
            'dataloader_path' : '/opt/airflow/dags/data/link_title_dataloader.pickle',
            'which_emb': 'linktitle_emb',
            'link_rec_on': False,
            'device' : 'cuda'  ##'cuda'
        }
    )


    
    task_calculate_piktitle_emb_vector = PythonOperator(
        task_id="calculate_piktitle_emb_vector",
        python_callable=calculate_emb_vectors.calculate_emb,
        op_kwargs={
            'default_path' : '/opt/airflow/dags',
            "processed_data_path": '/opt/airflow/dags/data/processed_data.csv', 
            'tokenizer_name': 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
            'model_name': 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
            'dataloader_path' : '/opt/airflow/dags/data/pik_title_dataloader.pickle',
            'which_emb': 'piktitle_emb',
            'link_rec_on': False,
            'device' : 'cuda'  ##'cuda'
        }
    )
    
    

    
    task_make_bento_model = PythonOperator(
        task_id="make_bento_model",
        python_callable=make_bento_model.make_bento_model,
        op_kwargs={
            "model_name": 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking', 
            'tokenizer_name': 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
            'huggingface_pipeline_name': 'feature-extraction',
            'bentoml_model_name' : 'pik_recommender_model',
        }
    )
    
    
    
    
    task_create_bento = BashOperator(   ##이전 bentoml model들을 모두 지운다
        task_id="create_bento",         ##
        bash_command=
        
        """ 
            cd /opt/airflow/dags/bentoml/pik_rec_bento; bentoml build
        """
        )
    
        
        
    
    
    task_serve_bentoml = BashOperator(
        task_id="serve_bentoml",
        bash_command=
        # fuser -k 3000/tcp; bentoml serve -p 3001 pik_recommender_bento:latest --production
        """ 
            
            
            fuser -k 3001/tcp; bentoml serve -p 3001 pik_recommender_bento:latest --production
        """
        )
        
    
    



    task_clear_bento >> task_data_process >> task_save_processed_data >> task_linktitle_data_process_to_torch >>  task_piktitle_data_process_to_torch >> task_calculate_linktitle_emb_vector_and_pik_vector >> task_calculate_piktitle_emb_vector >> task_make_bento_model >> task_create_bento >> task_serve_bentoml

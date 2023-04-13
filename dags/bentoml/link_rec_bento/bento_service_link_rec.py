#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:59:38 2022

@author: hojun
"""

from random import random
import torch
from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel, AutoTokenizer, AutoModel, Trainer, TrainingArguments
import bentoml
import numpy as np
from bentoml.io import Text, JSON, NumpyNdarray, Multipart, File, PandasDataFrame
import typing
from sklearn.metrics.pairwise import cosine_similarity
import pydantic
import requests_toolbelt
import pandas as pd
import json
import pickle
from typing import Any
import io
import lshashpy3
import fasttext
import re
from sentence_transformers import SentenceTransformer


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)


model_name = 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking'
# model = bentoml.transformers.load_runner(model_name)#.to('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# runner = bentoml.transformers.get("feature-extraction:latest").to_runner()
runner = bentoml.transformers.get("link_recommender_model:latest").to_runner()

# runner.init_local()
# svc = bentoml.Service("feature-extraction-service", runners=[runner])
svc = bentoml.Service("link_recommender_bento", runners=[runner])

# model = SentenceTransformer('sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking')


# lsh = lshashpy3.LSHash(hash_size=70, input_dim=768, num_hashtables=35,
#     storage_config={ 'dict': None },
#     matrices_filename='/opt/airflow/dags/data/lsh_matrices.npz',
#     hashtable_filename='/opt/airflow/dags/data/lsh_hashtables.npz',
#     overwrite=False)


with open("/opt/airflow/dags/data/linktitle_emb_vec.json") as f:  ##avg_{data_type}_vec.pickle  avg_pik_vec.json or avg_user_vec.json
    links_vec = json.load(f)

with open("/opt/airflow/dags/data/pik_vec.json") as f:  ##avg_{data_type}_vec.pickle  avg_pik_vec.json or avg_user_vec.json
    piks_vec = json.load(f)
    
with open("/opt/airflow/dags/data/piktitle_emb_vec.json") as f:  ##pik_id_embeddings_vectors.json
    piktitle_vec = json.load(f)
    
with open("/opt/airflow/dags/data/num_link_by_pik.json") as f:
    num_link_by_pik = json.load(f)

with open("/opt/airflow/dags/data/user_lang_dict_userset.json") as f:
    user_lang_dict_userset = json.load(f)
 
with open("/opt/airflow/dags/data/pik_lang_dict_userset.json") as f:
    pik_lang_dict_userset = json.load(f)

with open("/opt/airflow/dags/data/link_lang_dict_userset.json") as f:
    link_lang_dict_userset = json.load(f)

with open("/opt/airflow/dags/data/user_lang_dict_detected.json") as f:
    user_lang_dict_detected = json.load(f)
 
with open("/opt/airflow/dags/data/pik_lang_dict_detected.json") as f:
    pik_lang_dict_detected = json.load(f)
    
with open("/opt/airflow/dags/data/link_lang_dict_detected.json") as f:
    link_lang_dict_detected = json.load(f)
    
with open('/opt/airflow/dags/data/pik_link.json') as f:
    pik_link = json.load(f)

with open('/opt/airflow/dags/data/user_pik.json') as f:
    user_pik = json.load(f)
    
with open('/opt/airflow/dags/data/user_link.json') as f:
    user_link = json.load(f)
    
with open('/opt/airflow/dags/data/pik_status_dict.json') as f:
    pik_status_dict = json.load(f)

with open('/opt/airflow/dags/data/link_status_dict.json') as f:
    link_status_dict = json.load(f)

with open('/opt/airflow/dags/data/linkid_title_dict.json') as f:
    linkid_title_dict = json.load(f)

with open('/opt/airflow/dags/data/pikid_title_dict.json') as f:
    pikid_title_dict = json.load(f)

with open("/opt/airflow/dags/data/link_title_vec.json") as f:  ##avg_{data_type}_vec.pickle  avg_pik_vec.json or avg_user_vec.json
    link_title_vec = json.load(f)

with open("/opt/airflow/dags/data/link_description_vec.json") as f:  ##avg_{data_type}_vec.pickle  avg_pik_vec.json or avg_user_vec.json
    link_description_vec = json.load(f)

with open("/opt/airflow/dags/data/pik_title_vec.json") as f:  ##pik_id_embeddings_vectors.json
    pik_title_vec = json.load(f)

with open("/opt/airflow/dags/data/cat_title_vec.json") as f:  ##pik_id_embeddings_vectors.json
    cat_title_vec = json.load(f)


path_to_pretrained_model = '/opt/airflow/dags/data/lid.176.bin'

fmodel = fasttext.load_model(path_to_pretrained_model)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking')


processed_data = pd.read_csv('/opt/airflow/dags/data/processed_data.csv', lineterminator='\n')



from tqdm import tqdm
pik_title_vec = {int(k) : np.array(v) for k, v in pik_title_vec.items()}
link_title_vec = {int(k) : np.array(v) for k, v in link_title_vec.items()}
cat_title_vec = {int(k) : np.array(v) for k, v in cat_title_vec.items()}
link_description_vec = {int(k) : np.array(v) for k, v in link_description_vec.items()}




processed_data['link_title_vec'] = processed_data['link_id'].map(link_title_vec)
processed_data['link_description_vec']= processed_data['link_id'].map(link_description_vec)
processed_data['pik_title_vec'] = processed_data['pik_id'].map(pik_title_vec)
processed_data['cat_title_vec'] = processed_data['cat_id'].map(cat_title_vec)


all_vecs_dict = {}
for index, row in tqdm(processed_data[['link_title_vec', 'link_description_vec', 'pik_title_vec']].iterrows()):
    all_vecs_dict[index] = (row['link_title_vec'] + row['link_description_vec'] + row['pik_title_vec'])/len(row)
    
    

processed_data['final_vec'] = processed_data.index.map(all_vecs_dict)     
    




##when loading pretrained one.
lsh = lshashpy3.LSHash(hash_size=60, input_dim=768, num_hashtables=60,
    storage_config={ 'dict': None },
    matrices_filename='/opt/airflow/dags/data/new_lsh_matrices.npz',
    hashtable_filename='/opt/airflow/dags/data/new_lsh_hashtables.npz',
    overwrite=False)


























def preprocess(document: str) -> str:
    
    ##https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=realuv&logNo=220699272999
    pattern =r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9 ]' 
    # pattern =r'[^가-힣ㄱ-ㅎㅏ-ㅣ]' ##숫자와 영어를뺴고싶은경우
    document = re.sub(pattern=pattern, repl=' ', string=document)
    
    # 영어 소문자로 변환
    document = document.lower()

    # remove empty space
    document = document.strip()

    # make empty spcae size as only one
    document = ' '.join(document.split())
    
    return document


def get_most_similar_links_lsh_en(link_id, processed_data, link_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold):    
    '''
    lsh_result[0][0][1] ## similarity
    lsh_result[0][1] ## link_id
    '''
    lsh.index(np.array(processed_data['final_vec'][processed_data['link_id'] == int(link_id)][0]), extra_data = processed_data['link_id'][processed_data['link_id']== 1379][0])
    lsh_result = lsh.query(np.array(processed_data['final_vec'][processed_data['link_id'] == int(link_id)][0]) , num_results=lsh_topk, distance_func="cosine")
    # lsh.index(links_vec[link_id], extra_data = int(link_id))
    # lsh_result = lsh.query(links_vec[link_id] , num_results=lsh_topk, distance_func="cosine")
    # lsh_result = lsh.query(links_vec[link_id] , num_results=10, distance_func="cosine")
    lsh_result = sorted(lsh_result, key=lambda x: x[1], reverse=False)
    
    lsh_result_dict = {}
    for single_result in lsh_result:
        lsh_result_dict[single_result[0][1]] = single_result[1]

    filtered_result_dict = {}
    filtered_result_list = []
    for link_rec in lsh_result:
        if link_status_dict[str(link_rec[0][1])] == 'public':## 공개된 링크이고 현재픽이 아니라면
            if link_id in link_status_dict.keys():
                if link_lang_dict_detected[str(link_rec[0][1])] == 'en' and (link_lang_dict_detected[str(link_rec[0][1])] != 'ko' or link_lang_dict_detected[str(link_rec[0][1])] != 'kr'): ##link_rec[0][0][1] = link_id
                    # thisSim = cosine_similarity(np.array(vec).reshape(1, -1), np.array(piks_vec[pik_id]).reshape(1, -1))
                    filtered_result_dict[link_rec[0][1]] = (link_rec[1], link_rec[0][0]) ##link_rec[0][1]=link_id, link_rec[1]=cosine_distance, link_rec[0][0]=embedding
                    filtered_result_list.append((link_rec[0][1], link_rec[1], link_rec[0][0]))
    
    query_link_cos_dis = round(lsh_result_dict[int(link_id)], 6)  ##쿼리 링크의 cosine distance
                    
    extra_filtered_result = []
    for i, (lid, (cos_distance, embedding)) in enumerate(filtered_result_dict.items()):
        if cos_distance < threshold:
                
                # extra_filtered_result.append((lid, cos_distance))
            if i == 0 and round(filtered_result_list[i][1], 6) != query_link_cos_dis:
                extra_filtered_result.append((lid, cos_distance))

            
            if i >= 1:    
                if round(filtered_result_list[i][1], 6) != round(filtered_result_list[i-1][1], 6) and round(filtered_result_list[i][1], 6) != query_link_cos_dis: ##유사도가 바로그다음으로높은것과비교했을때 현재유사도와같으면 같은내용의픽이나링크일테니 그건스킵하라는것
                    extra_filtered_result.append((lid, cos_distance))
                    if len(extra_filtered_result) == final_top_k:
                            break
                        
    return extra_filtered_result




def get_most_similar_links_lsh_ko(link_id, processed_data, link_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold):
    
    '''
    lsh_result[0][0][1] ## similarity
    lsh_result[0][1] ## link_id
    '''
    # import time
    # s=time.time()
    ##실시간 LSH쿼리를 하기위해서는 실시간으로 이미 트레이닝된 lsh테이블에 쿼리를 할 것도 인덱싱해줘야 그다음 쿼리를 할 수 있다
    lsh.index(np.array(processed_data['final_vec'][processed_data['link_id'] == int(link_id)][0]), extra_data = processed_data['link_id'][processed_data['link_id']== 1379][0])
    lsh_result = lsh.query(np.array(processed_data['final_vec'][processed_data['link_id'] == int(link_id)][0]) , num_results=lsh_topk, distance_func="cosine")
    # lsh_result = lsh.query(links_vec[link_id] , num_results=10, distance_func="cosine")
    lsh_result = sorted(lsh_result, key=lambda x: x[1], reverse=False)
    # print(time.time()-s)
    
    lsh_result_dict = {}
    for single_result in lsh_result:
        lsh_result_dict[single_result[0][1]] = single_result[1]
    # print(time.time()-s)

    filtered_result_dict = {}
    filtered_result_list = []
    
    for link_rec in lsh_result:
        if link_status_dict[str(link_rec[0][1])] == 'public':## 공개된 링크이고 현재픽이 아니라면
            if link_id in link_status_dict.keys():
                if link_lang_dict_detected[str(link_rec[0][1])] == 'ko' or link_lang_dict_detected[str(link_rec[0][1])] == 'kr': ##link_rec[0][0][1] = link_id
                ### if link_lang_dict_detected[str(link_rec[0][1])] == 'ko' or link_lang_dict_detected[str(link_rec[0][1])] == 'kr': ##link_rec[0][0][1] = link_id
                    filtered_result_dict[link_rec[0][1]] = (link_rec[1], link_rec[0][0]) ##link_rec[0][1]=link_id, link_rec[1]=cosine_distance, link_rec[0][0]=embedding
                    filtered_result_list.append((link_rec[0][1], link_rec[1], link_rec[0][0]))

    query_link_cos_dis = round(lsh_result_dict[int(link_id)], 6)  ##쿼리 링크의 cosine distance
                
    extra_filtered_result = []
    for i, (lid, (cos_distance, embedding)) in enumerate(filtered_result_dict.items()):
        if cos_distance < threshold:
            
                # extra_filtered_result.append((lid, cos_distance))
            if i == 0 and round(filtered_result_list[i][1], 6) != query_link_cos_dis:
                extra_filtered_result.append((lid, cos_distance))
            
            if i >= 1:    
                if round(filtered_result_list[i][1], 6) != round(filtered_result_list[i-1][1], 6) and round(filtered_result_list[i][1], 6) != query_link_cos_dis: ##유사도가 바로그다음으로높은것과비교했을때 현재유사도와같으면 같은내용의픽이나링크일테니 그건스킵하라는것
                    extra_filtered_result.append((lid, cos_distance))
                    if len(extra_filtered_result) == final_top_k:
                            break
                        
    return extra_filtered_result
 
                
       
                
def get_most_similar_links_online_lsh_en(link_text, user_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold):    
    '''
    lsh_result[0][0][1] ## similarity
    lsh_result[0][1] ## link_id
    '''
    processed_link_text = preprocess(link_text)
    # lang_pred_link = fmodel.predict([processed_link_text])[0][0][0][-2:]
    embeddings = model.encode(processed_link_text)
    
    lsh.index(embeddings)
    lsh_result = lsh.query(embeddings , num_results=lsh_topk, distance_func="cosine")
    lsh_result = sorted(lsh_result, key=lambda x: x[1], reverse=False)

    filtered_result_dict_en = {}
    filtered_result_list_en = []
         
    for link_rec in lsh_result: ##지금 실시간으로 들어온 값은 link_id까 없기 때문에 None일 것이다. 그러므로 이것이 나오면 사뿐히 제껴주자
        if link_rec[0][1] is None: ##link_rec[0][1] link_id
            query_link_cos_dis = abs(round(link_rec[1], 6))
            continue

        if link_status_dict[str(link_rec[0][1])] == 'public':## 공개된 링크이고 현재픽이 아니라면
            if link_lang_dict_detected[str(link_rec[0][1])] == 'en' and (link_lang_dict_detected[str(link_rec[0][1])] != 'ko' or link_lang_dict_detected[str(link_rec[0][1])] != 'kr'): ##link_rec[0][0][1] = link_id
                # thisSim = cosine_similarity(np.array(vec).reshape(1, -1), np.array(piks_vec[pik_id]).reshape(1, -1))
                filtered_result_dict_en[link_rec[0][1]] = (link_rec[1], link_rec[0][0]) ##link_rec[0][1]=link_id, link_rec[1]=cosine_distance, link_rec[0][0]=embedding
                filtered_result_list_en.append((link_rec[0][1], link_rec[1], link_rec[0][0]))
      
                
    extra_filtered_result_en = []
    for i, (lid, (cos_distance, embedding)) in enumerate(filtered_result_dict_en.items()):
        if cos_distance < threshold:
            if i == 0 and query_link_cos_dis != abs(round(cos_distance, 6)):
                extra_filtered_result_en.append((lid, cos_distance))
            if i >= 1:    
                if abs(round(filtered_result_list_en[i][1], 6)) != abs(round(filtered_result_list_en[i-1][1], 6)) and abs(round(filtered_result_list_en[i][1], 6)) != abs(round(query_link_cos_dis, 6)): ##유사도가 바로그다음으로높은것과비교했을때 현재유사도와같으면 같은내용의픽이나링크일테니 그건스킵하라는것
                    extra_filtered_result_en.append((lid, cos_distance))
                    if len(extra_filtered_result_en) == final_top_k:
                            break
                        
    return extra_filtered_result_en
# 
# zz = get_most_similar_links_online_lsh_en(link_text, user_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold)                   
 
# processed_data['link_title'][processed_data['link_id'] == 120478] 




def get_most_similar_links_online_lsh_ko(link_text, user_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold):    
    '''
    lsh_result[0][0][1] ## similarity
    lsh_result[0][1] ## link_id
    '''
    processed_link_text = preprocess(link_text)
    embeddings = model.encode(processed_link_text)
    
    lsh.index(embeddings)
    lsh_result = lsh.query(embeddings , num_results=lsh_topk, distance_func="cosine")
    lsh_result = sorted(lsh_result, key=lambda x: x[1], reverse=False)

    filtered_result_dict_ko = {}
    filtered_result_list_ko = []
        

    for link_rec in lsh_result: ##지금 실시간으로 들어온 값은 link_id까 없기 때문에 None일 것이다. 그러므로 이것이 나오면 사뿐히 제껴주자
        if link_rec[0][1] is None:
            query_link_cos_dis = abs(round(link_rec[1], 6))
            continue

        elif link_status_dict[str(link_rec[0][1])] == 'public':## 공개된 링크이고 현재픽이 아니라면
            if link_lang_dict_detected[str(link_rec[0][1])] == 'ko' or link_lang_dict_detected[str(link_rec[0][1])] == 'kr': ##link_rec[0][0][1] = link_id
                # thisSim = cosine_similarity(np.array(vec).reshape(1, -1), np.array(piks_vec[pik_id]).reshape(1, -1))
                filtered_result_dict_ko[link_rec[0][1]] = (link_rec[1], link_rec[0][0]) ##link_rec[0][1]=link_id, link_rec[1]=cosine_distance, link_rec[0][0]=embedding
                filtered_result_list_ko.append((link_rec[0][1], link_rec[1], link_rec[0][0]))
                

                 
    extra_filtered_result_ko = []
    for i, (lid, (cos_distance, embedding)) in enumerate(filtered_result_dict_ko.items()):
        if cos_distance < threshold:
            if i == 0 and query_link_cos_dis != abs(round(cos_distance, 6)):
                extra_filtered_result_ko.append((lid, cos_distance))
            if i >= 1:    
                if abs(round(filtered_result_list_ko[i][1], 6)) != abs(round(filtered_result_list_ko[i-1][1], 6)) and abs(round(filtered_result_list_ko[i][1], 6)) != abs(round(query_link_cos_dis, 6)): ##유사도가 바로그다음으로높은것과비교했을때 현재유사도와같으면 같은내용의픽이나링크일테니 그건스킵하라는것
                    extra_filtered_result_ko.append((lid, cos_distance))
                    if len(extra_filtered_result_ko) == final_top_k:
                            break
                        
    return extra_filtered_result_ko


    
        
        
        
        
        
        
        
        
        
        
        
        
        
def rec_link_by_lang(link_id, user_id, link_text, processed_data, link_status_dict, link_lang_dict_detected, user_lang_dict_detected, lsh, lsh_topk, final_top_k, threshold):    
    # try:
        # if pd.unique(data['language_code'][data['user_id'] == int(user_id)])[0] == 'ko': ##language_cde가 'ko' 인지, 'en'인지,
    if user_id in user_link.keys(): ##등록되어있는 유저들 중에,
        if link_id in link_lang_dict_detected.keys(): ##링크도 이미 등록되어 있다면
            if (user_lang_dict_detected[user_id] == 'ko' or user_lang_dict_detected[user_id] == 'kr'):
                print('유저 존재하고 링크 존재하고 한국어 추천을 한다')
                '''
                유저의 링크들의 합이 가장 많은 수가 한국어 일 때 추천해 는 로직 
                '''
                result = get_most_similar_links_lsh_ko(link_id, processed_data, link_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold)
                return result
            
            elif user_lang_dict_detected[user_id] == 'en' and (user_lang_dict_detected[user_id] != 'ko' or user_lang_dict_detected[user_id] != 'kr'):
                print('유저 존재하고 링크 존재하고 영어 추천을 한다')
                result = get_most_similar_links_lsh_en(link_id, processed_data, link_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold)
                return result
            

        elif link_id not in link_lang_dict_detected.keys(): ##링크가 아직 등록되지 않았다면,
            '''
            실시간으로 링크정보를 가져와 프로세싱하고 LSH한 후 바로 추천해 줄 수 있도록 한다.
            '''
            if (user_lang_dict_detected[user_id] == 'ko' or user_lang_dict_detected[user_id] == 'kr'):
                print('유저 존재하지만 링크 존재하지 않고 한국어 추천을 한다')
                result = get_most_similar_links_online_lsh_ko(link_text, link_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold)
                return result
            
            elif user_lang_dict_detected[user_id] == 'en' and (user_lang_dict_detected[user_id] != 'ko' or user_lang_dict_detected[user_id] != 'kr'):
                print('유저 존재하지만 링크 존재하지 않고 영어 추천을 한다')
                result = get_most_similar_links_online_lsh_en(link_text, link_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold)
                return result
           
            
           
            

    elif user_id not in user_link.keys(): ##유저 마저도 추천시스템 정보에 등록이 안된 새 유저라면,
        '''
        만약에 유저가 추천시스템에 아직 등록이 안된 유저여도 클릭한 링크의 언어를 베이스로 추천해줄 수 있도록 하는 방식
        '''
        
        if link_id in link_lang_dict_detected.keys(): ##링크도 이미 등록되어 있다면
            if (link_lang_dict_detected[link_id] == 'ko' or link_lang_dict_detected[link_id] == 'kr'):
                print('유저 존재하지 않지만 링크는 존재하며 한국어 추천을 한다')
                result = get_most_similar_links_lsh_ko(link_id, processed_data, link_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold)
                return result
            
            elif link_lang_dict_detected[link_id] == 'en' and (link_lang_dict_detected[link_id] != 'ko' or link_lang_dict_detected[link_id] != 'kr'):
                print('유저 존재하지 않지만 링크는 존재하며 영어 추천을 한다')
                result = get_most_similar_links_lsh_en(link_id, processed_data, link_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold)
                return result
            

        elif link_id not in link_lang_dict_detected.keys(): ##링크가 아직 등록되지 않았다면,

            processed_link_text = preprocess(link_text)
            lang_pred_link = fmodel.predict([processed_link_text])[0][0][0][-2:]
            
            if (lang_pred_link == 'ko' or lang_pred_link == 'kr'):
                print('유저가 존재하지 않고 링크도 존재하지 않으며 한국어 추천을 한다')    
                result = get_most_similar_links_online_lsh_ko(link_text, link_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold)
                return result
            
            elif lang_pred_link == 'en' and (lang_pred_link != 'ko' or lang_pred_link != 'kr'):
                print('유저가 존재하지 않고 링크도 존재하지 않으며 영어 추천을 한다')    
                result = get_most_similar_links_online_lsh_en(link_text, link_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold)
                return result    
    
        
        return None


# zz = rec_link_by_lang('1379', '17', 'hh', link_status_dict, link_lang_dict_detected, user_lang_dict_detected, lsh, lsh_topk, final_top_k, threshold)
# z = processed_data[processed_data['link_id'] == 1379]


# xx = rec_link_by_lang('1379', '32423532525235', 'hh', link_status_dict, link_lang_dict_detected, user_lang_dict_detected, lsh, lsh_topk, final_top_k, threshold)

# cc = rec_link_by_lang('10664', '32423532525235', 'hh', link_status_dict, link_lang_dict_detected, user_lang_dict_detected, lsh, lsh_topk, final_top_k, threshold)
# z = processed_data[processed_data['link_id'] == 10664]
# processed_data['link_title'][processed_data['link_id'] == 10666]
# processed_data['link_title'][processed_data['link_id'] == 65626]
# processed_data['link_title'][processed_data['link_id'] == 86093]
# processed_data['link_title'][processed_data['link_id'] == 86197]
# get_most_similar_links_lsh_ko('10664', links_vec, link_lang_dict_detected, link_status_dict, lsh, lsh_topk, final_top_k, threshold)

# vv = rec_link_by_lang('12342340664', '32423532525235', 'How to become a marketer', link_status_dict, link_lang_dict_detected, user_lang_dict_detected, lsh, lsh_topk, final_top_k, threshold)
# processed_data['link_title'][processed_data['link_id'] == 260]
# processed_data['link_title'][processed_data['link_id'] == 559052]

# bb = rec_link_by_lang('12342340664', '32423532525235', '여행', link_status_dict, link_lang_dict_detected, user_lang_dict_detected, lsh, lsh_topk, final_top_k, threshold)
# processed_data['link_title'][processed_data['link_id'] == 123614]
# processed_data['link_title'][processed_data['link_id'] == 86942]
# processed_data['link_title'][processed_data['link_id'] == 42607]
# processed_data['link_title'][processed_data['link_id'] == 91830]


## This is modification of above to inlcude language condition
input_spec = Multipart(user_id=Text(), link_id=Text(), link_text=Text())
@svc.api(input=input_spec, output=JSON())
async def predict(user_id, link_id, link_text) -> dict:
    similarity_list = await rec_link_by_lang(link_id, user_id, link_text, link_status_dict, link_lang_dict_detected, user_lang_dict_detected, lsh, lsh_topk=50, final_top_k=5, threshold=0.3)
    similarity_dict = {similarity_pair[0]: similarity_pair[1] for similarity_pair in similarity_list} ##dict {user: similarity}   
    return sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)





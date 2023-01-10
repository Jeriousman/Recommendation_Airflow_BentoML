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

model_name = 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking'
# model = bentoml.transformers.load_runner(model_name)#.to('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# runner = bentoml.transformers.get("feature-extraction:latest").to_runner()
runner = bentoml.transformers.get("link_recommender_model:latest").to_runner()

# runner.init_local()
# svc = bentoml.Service("feature-extraction-service", runners=[runner])
svc = bentoml.Service("link_recommender_bento", runners=[runner])




with open("/opt/airflow/dags/data/linktitle_emb_vec.json") as f:  ##avg_{data_type}_vec.pickle  avg_pik_vec.json or avg_user_vec.json
    link_vec = json.load(f)


with open("/opt/airflow/dags/data/pik_vec.json") as f:  ##avg_{data_type}_vec.pickle  avg_pik_vec.json or avg_user_vec.json
    piks_vec = json.load(f)
    
with open("/opt/airflow/dags/data/piktitle_emb_vec.json") as f:  ##pik_id_embeddings_vectors.json
    piktitle_vec = json.load(f)
    
    
with open("/opt/airflow/dags/data/num_link_by_pik.json") as f:
    num_link_by_pik = json.load(f)

with open("/opt/airflow/dags/data/num_link_by_user.json") as f:
    num_link_by_pik = json.load(f)

# df = pd.read_csv("/opt/airflow/dags/data/processed_data.csv")

 
with open("/opt/airflow/dags/data/user_lang_dict.json") as f:
    user_lang_dict = json.load(f)

with open("/opt/airflow/dags/data/pik_lang_dict.json") as f:
    pik_lang_dict = json.load(f)

with open("/opt/airflow/dags/data/link_lang_dict.json") as f:
    link_lang_dict = json.load(f)


with open('/opt/airflow/dags/data/user_pik.json') as f:
    user_pik = json.load(f)

with open('/opt/airflow/dags/data/user_link.json') as f:
    user_link = json.load(f)






lsh = lshashpy3.LSHash(hash_size=20, input_dim=768, num_hashtables=10,
    storage_config={ 'dict': None },
    matrices_filename='/opt/airflow/dags/data/lsh_matrices_filename.npz',
    hashtable_filename='/opt/airflow/dags/data/lsh_hashtables_filename.npz',
    overwrite=False)







# with open("/home/hojun/airflow/dags/data/linktitle_emb_vec.json") as f:  ##avg_{data_type}_vec.pickle  avg_pik_vec.json or avg_user_vec.json
#     link_vec = json.load(f)


# with open("/home/hojun/airflow/dags/data/pik_vec.json") as f:  ##avg_{data_type}_vec.pickle  avg_pik_vec.json or avg_user_vec.json
#     piks_vec = json.load(f)
    
# with open("/home/hojun/airflow/dags/data/piktitle_emb_vec.json") as f:  ##pik_id_embeddings_vectors.json
#     piktitle_vec = json.load(f)
    
    
# with open("/home/hojun/airflow/dags/data/num_link_by_pik.json") as f:
#     num_link_by_pik = json.load(f)

# with open("/home/hojun/airflow/dags/data/num_link_by_user.json") as f:
#     num_link_by_pik = json.load(f)

# df = pd.read_csv("/home/hojun/airflow/dags/data/processed_data.csv")

 
# with open("/home/hojun/airflow/dags/data/user_lang_dict.json") as f:
#     user_lang_dict = json.load(f)

# with open("/home/hojun/airflow/dags/data/pik_lang_dict.json") as f:
#     pik_lang_dict = json.load(f)

# with open("/home/hojun/airflow/dags/data/link_lang_dict.json") as f:
#     link_lang_dict = json.load(f)


# with open('/home/hojun/airflow/dags/data/user_pik.json') as f:
#     user_pik = json.load(f)

# with open('/home/hojun/airflow/dags/data/user_link.json') as f:
#     user_link = json.load(f)





# lsh = lshashpy3.LSHash(hash_size=20, input_dim=768, num_hashtables=10,
#     storage_config={ 'dict': None },
#     matrices_filename='/home/hojun/airflow/dags/data/lsh_matrices_filename.npz',
#     hashtable_filename='/home/hojun/airflow/dags/data/lsh_hashtables_filename.npz',
#     overwrite=False)



# df['link_title'][df['link_id'] == 1407]
# df['link_title'][df['link_id'] == 1407]
# df['link_title'][df['link_id'] == 1407]
# df['link_title'][df['link_id'] == 1407]


# user_id = '18'
# link_id = '1407'
# link_vector = link_vec
# lsh_table = lsh
# lsh_topk=40
# topk = 10
# data=link_cat_pik




def get_most_similar_links_lsh(link_id, user_id, link_vector, user_link, lsh, lsh_topk, topk):
    first_result = lsh.query(link_vector[link_id] , num_results=lsh_topk, distance_func="cosine")
    sorted_result = sorted(first_result, key=lambda x: x[1], reverse=False)
    filtered_result = []
    for i, result in enumerate(sorted_result):
        if int(link_id) !=  result[0][1] and int(result[0][1]) not in user_link[user_id]: ##링크가 본 유저에게 이미 속해있지 않고, 본 픽이 추천픽이 아닌 경우에 추천하라는 것        
        # if (link_vector[link_id] !=  sorted_result[i][0][0]).all(): ##본 링크가 아니라면 추천하라는 뜻 
            if i >= 1:  
                if sorted_result[i-1][0][0] !=  sorted_result[i][0][0]: ##전 링크가 지금링크와 엠베딩 값이 같으면 아마도 같은 것일것이기 때문에추천에서 빼라는뜻
                    # if i <= topk:
                    # if sorted_result[i][0][1]  != link_id:
                    filtered_result.append((sorted_result[i][0][1], sorted_result[i][1]))
                    if len(filtered_result) == topk:
                        break
                    # else:
                    #     break
                        
                        
    filtered_result = sorted(filtered_result, key=lambda x: x[1], reverse=False)


    ## 만약 sim_list가비어있거나 None이면, 그보다 더유사도가 적지만 그래도괜찮을수있는것을 추천해준다. 
    if filtered_result == None:
        print('There are no recommended links for your link now')
    
    elif bool(filtered_result):
        return  filtered_result    
    
                                    
    else:
        print('Hey, there are really not suitable recommendation for your link now. But we are working on it!')



def get_most_similar_links_lsh_ko(link_id, user_id, link_vector, user_link, link_lang_dict, lsh, lsh_topk, topk):
    
    first_result = lsh.query(link_vector[link_id] , num_results=lsh_topk, distance_func="cosine")
    sorted_result = sorted(first_result, key=lambda x: x[1], reverse=False)
    filtered_result = []
    for i, result in enumerate(sorted_result):
        if link_lang_dict[str(result[0][1])] == 'ko': ##language_cde가 'ko' 인지, 'en'인지. result[0][1]는 user_id이다. 예: 40
            if int(link_id) !=  result[0][1] and int(result[0][1]) not in user_link[user_id]: ##링크가 본 유저에게 이미 속해있지 않고, 본 픽이 추천픽이 아닌 경우에 추천하라는 것
            # if (link_vector[link_id] !=  sorted_result[i][0][0]).all(): ##본 링크가 아니라면 추천하라는 뜻 
                if i >= 1:  
                    if sorted_result[i-1][0][0] !=  sorted_result[i][0][0]: ##전 링크가 지금링크와 엠베딩 값이 같으면 아마도 같은 것일것이기 때문에추천에서 빼라는뜻
                        filtered_result.append((sorted_result[i][0][1], sorted_result[i][1]))
                        if len(filtered_result) == topk:
                            break
    
    filtered_result = sorted(filtered_result, key=lambda x: x[1], reverse=False)
    

    ## 만약 sim_list가비어있거나 None이면, 그보다 더유사도가 적지만 그래도괜찮을수있는것을 추천해준다. 
    if filtered_result == None:
        print('죄송하지만 추천 할 만한 링크가 없어요. 더 노력할게요!')
    
    elif bool(filtered_result):
        return  filtered_result    
    

                                    
    else:
        print('죄송해요! 이 링크에 맞는 추천 할 만한 링크가 없어요 :( 열심히 노력해서 더 좋은 추천서비스를 제공할게요!')







def get_most_similar_links_lsh_en(link_id, user_id, link_vector, user_link, link_lang_dict, lsh, lsh_topk, topk):
    
    first_result = lsh.query(link_vector[link_id] , num_results=lsh_topk, distance_func="cosine")
    sorted_result = sorted(first_result, key=lambda x: x[1], reverse=False)
    filtered_result = []
    for i, result in enumerate(sorted_result):
        if link_lang_dict[str(result[0][1])] == 'en': ##language_cde가 'ko' 인지, 'en'인지. result[0][1]는 user_id이다. 예: 40
            if int(link_id) !=  result[0][1] and int(result[0][1]) not in user_link[user_id]: ##링크가 본 유저에게 이미 속해있지 않고, 본 픽이 추천픽이 아닌 경우에 추천하라는 것
                if i >= 1:  
                    if sorted_result[i-1][0][0] !=  sorted_result[i][0][0]: ##전 링크가 지금링크와 엠베딩 값이 같으면 아마도 같은 것일것이기 때문에추천에서 빼라는뜻
                        # if sorted_result[i][0][1]  != link_id:
                        filtered_result.append((sorted_result[i][0][1], sorted_result[i][1]))
                        if len(filtered_result) == topk:
                            break
    
    filtered_result = sorted(filtered_result, key=lambda x: x[1], reverse=False)
    

    ## 만약 sim_list가비어있거나 None이면, 그보다 더유사도가 적지만 그래도괜찮을수있는것을 추천해준다. 
    if filtered_result == None:
        print('There are no recommended links for your link now')
    
    elif bool(filtered_result):
        return  filtered_result    

                                    
    else:
        print('Hey, there are really not suitable recommendation for your link now. But we are working on it!')



# link_id = '599657'
# user_id = '7443'

def rec_link_by_lang(link_id, user_id, link_vector, user_link, user_lang_dict, link_lang_dict, lsh, lsh_topk=40, topk=10):    
    if user_lang_dict[user_id] == 'ko' or user_lang_dict[user_id] == 'kr': ##language_cde가 'ko' 인지, 'en'인지,
        result = get_most_similar_links_lsh_ko(link_id, user_id, link_vector, user_link, link_lang_dict, lsh, lsh_topk=40, topk=10)
        return result
    elif user_lang_dict[user_id] == 'en': ##language_cde가 'ko' 인지, 'en'인지,
        result = get_most_similar_links_lsh_en(link_id, user_id, link_vector, user_link, link_lang_dict, lsh, lsh_topk=40, topk=10)
        return result
    else: 
        result = get_most_similar_links_lsh(link_id, user_id, link_vector, user_link, lsh, lsh_topk=40, topk=10)
        return result
    

# link_id = '599657'
# user_id = '7443'


# df['link_title'][df['link_id'] == 599657]
# df['link_title'][df['link_id'] == 308336]
# df['link_title'][df['link_id'] == 10929]
# df['link_title'][df['link_id'] == 495362]
# df['link_title'][df['link_id'] == 308877]
# df['language_code'][df['link_id'] == 599657]
# df['language_code'][df['link_id'] == 308336]



# link_id = '597315'
# user_id = '7380'
# df['link_title'][df['link_id'] == 597315]
# df['link_title'][df['link_id'] == 69110]
# df['link_title'][df['link_id'] == 578177]
# df['link_title'][df['link_id'] == 80966]
# df['link_title'][df['link_id'] == 122353]
# df['language_code'][df['link_id'] == 597315]
# df['language_code'][df['link_id'] == 308336]



# link_id = '599657'
# user_id = '7443'





## This is modification of above to inlcude language condition
input_spec = Multipart(user_id=Text(), link_id=Text())
@svc.api(input=input_spec, output=JSON())
def predict(user_id, link_id) -> dict:
    similarity_list = rec_link_by_lang(link_id, user_id, link_vec, user_link, user_lang_dict, link_lang_dict, lsh, lsh_topk=40, topk=10)
    similarity_dict = {similarity_pair[0]: similarity_pair[1] for similarity_pair in similarity_list} ##dict {user: similarity}   
    return sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)

















# '''link recommender logic. But we will comment this out bcuz we will use locality sensitive hashing technique'''
# def get_most_similar_links(link_id, link_vec, topk=10, threshold=0.60, second_threshold=0.50):
#     sim = list()
        
#     for lid, vec in link_vec.items():

#         thisSim = cosine_similarity(np.array(vec).reshape(1, -1), np.array(link_vec[link_id]).reshape(1, -1))
#         sim.append((lid, thisSim[0][0]))
    
#     full_ranked_similar_items = sorted(sim, key=lambda x: x[1], reverse=True) ##full similarity list
#     ranked_similar_items = full_ranked_similar_items[:topk+1] ##only topk similarity list. 본픽도 들어가있기떄문에+1을해준다
     
    
#     sim_list = [] ##추천 candidate 추려내서 저장하는 리스트 
#     for i in range(1, topk+1):
#         if ranked_similar_items[topk][1] > 0.60:    
#             if i >= 1:  
#                 if ranked_similar_items[i-1][1] != ranked_similar_items[i][1]: ##유사도가 바로그다음으로높은것과비교했을때 현재유사도와같으면 같은내용의픽이나링크일테니 그건스킵하라는것
#                     sim_list.append((ranked_similar_items[i][0], ranked_similar_items[i][1]))
 
    
#     ## 만약 sim_list가비어있거나 None이면, 그보다 더유사도가 적지만 그래도괜찮을수있는것을 추천해준다. 
#     if sim_list == None:
#         print('There are no recommended links for your link now')
    
#     elif bool(sim_list):
#         return  sorted(sim_list, key=lambda x: x[1], reverse=True)    
    
#     # elif not bool(sim_list):  
#     elif not bool(sim_list): ##sim_list에 아무것도 존재하지 않으면, second_threshold를 사용해서 더낮지만 그래도 차선인 추천을 해준다. 
                 
        
#         sim_list = []
#         for i in range(1, topk+1):
#             if ranked_similar_items[i][1] > second_threshold:    
#                 if i >= 1:  
#                     if ranked_similar_items[i-1][1] != ranked_similar_items[i][1]: ##유사도가 바로그다음으로높은것과비교했을때 현재유사도와같으면 같은내용의픽이나링크일테니 그건스킵하라는것
#                         # if random() >= 0.5:
#                         sim_list.append((ranked_similar_items[i][0], ranked_similar_items[i][1]))
            
#         return  sorted(sim_list, key=lambda x: x[1], reverse=True)
                                    
#     else:
#         print('Hey, there are really not suitable recommendation for your link now. But we are working on it!')
                    


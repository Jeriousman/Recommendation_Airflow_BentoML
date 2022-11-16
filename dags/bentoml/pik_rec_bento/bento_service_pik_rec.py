#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:59:38 2022

@author: hojun
"""
from random import choice
from random import random
from random import randint
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

model_name = 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking'
# model = bentoml.transformers.load_runner(model_name)#.to('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# runner = bentoml.transformers.get("feature-extraction:latest").to_runner()
runner = bentoml.transformers.get("pik_recommender_model:latest").to_runner()

# runner.init_local()
# svc = bentoml.Service("feature-extraction-service", runners=[runner])
svc = bentoml.Service("pik_recommender_bento", runners=[runner])


with open("/opt/airflow/dags/data/pik_vec.json") as f:  ##avg_{data_type}_vec.pickle  avg_pik_vec.json or avg_user_vec.json
    piks_vec = json.load(f)
    
with open("/opt/airflow/dags/data/piktitle_emb_vec.json") as f:  ##pik_id_embeddings_vectors.json
    piktitle_vec = json.load(f)
    
    
with open("/opt/airflow/dags/data/num_link_by_pik.json") as f:
    num_link_by_pik = json.load(f)

# df = pd.read_csv("/opt/airflow/dags/data/processed_data.csv")

 
with open("/opt/airflow/dags/data/user_lang_dict.json") as f:
    user_lang_dict = json.load(f)

 
with open("/opt/airflow/dags/data/pik_lang_dict.json") as f:
    pik_lang_dict = json.load(f)




with open('/opt/airflow/dags/data/user_pik.json') as f:
    user_pik = json.load(f)

# with open("/opt/airflow/dags/data/link_lang_dict.json") as f:
#     link_lang_dict = json.load(f)

# with open('/opt/airflow/dags/data/user_link.json') as f:
#     user_link = json.load(f)



# with open('/home/hojun/airflow/dags/data/user_pik.json') as f:
#     user_pik = json.load(f)

# with open("/home/hojun/airflow/dags/data/pik_vec.json") as f:  ##avg_{data_type}_vec.pickle  avg_pik_vec.json or avg_user_vec.json
#     piks_vec = json.load(f)
    
# with open("/home/hojun/airflow/dags/data/piktitle_emb_vec.json") as f:  ##pik_id_embeddings_vectors.json
#     piktitle_vec = json.load(f)
    
    
# with open("/home/hojun/airflow/dags/data/num_link_by_pik.json") as f:
#     num_link_by_pik = json.load(f)

# df = pd.read_csv("/home/hojun/airflow/dags/data/processed_data.csv")

 
# with open("/home/hojun/airflow/dags/data/user_lang_dict.json") as f:
#     user_lang_dict = json.load(f)

 
# with open("/home/hojun/airflow/dags/data/pik_lang_dict.json") as f:
#     pik_lang_dict = json.load(f)





def get_most_similar_piks(pik_id, user_id, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold):
    sim = list()
        
    for uid, vec in piks_vec.items():

        thisSim = cosine_similarity(np.array(vec).reshape(1, -1), np.array(piks_vec[pik_id]).reshape(1, -1))
        sim.append((uid, thisSim[0][0]))
    
    full_ranked_similar_items = sorted(sim, key=lambda x: x[1], reverse=True) ##full similarity list
    ranked_similar_items = full_ranked_similar_items[:topk+1] ##only topk similarity list. 본픽도 들어가있기떄문에+1을해준다
     
        
    # sim_dict = {} ##추천 candidate 추려내서 저장하는 딕셔너리 
    sim_list = [] 
    for i in range(0, topk+1):
        if ranked_similar_items[i][1] > threshold:   
            if int(pik_id) != int(ranked_similar_items[i][0]) and int(ranked_similar_items[i][0]) not in user_pik[user_id] and ranked_similar_items[i][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                if i >= 1:  
                    if ranked_similar_items[i-1][1] != ranked_similar_items[i][1]: ##유사도가 바로그다음으로높은것과비교했을때 현재유사도와같으면 같은내용의픽이나링크일테니 그건스킵하라는것
                        if num_link_by_pik[ranked_similar_items[i][0]] >= num_link_threshold: ##픽안에 num_link_threshold 갯수이상 링크가 존재할때만 추천한다 
                            lottery = random()
                            if lottery <= 0.7:
                                sim_list.append({'pik_id':ranked_similar_items[i][0], 'similarity':ranked_similar_items[i][1]})

                                
                            elif 0.7 < lottery <= 0.88:
                                pik_title_sim = cosine_similarity(np.array(piktitle_vec[pik_id]).reshape(1, -1), np.array(piktitle_vec[ranked_similar_items[i][0]]).reshape(1, -1))[0][0]
                                if pik_title_sim >= piktitle_threshold: ##픽타이틀의 유사도도 threshold를 넘으면 그대로바로 추천리스트에들어간다 
                                    sim_list.append({'pik_id':ranked_similar_items[i][0], 'similarity':ranked_similar_items[i][1]})    

                            
                            elif 0.88 < lottery <= 1.0:
      
                                random_topk_rec_index = randint(0, topk)
                                if int(pik_id) != int(ranked_similar_items[random_topk_rec_index][0]) and int(ranked_similar_items[random_topk_rec_index][0]) not in user_pik[user_id] and ranked_similar_items[random_topk_rec_index][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]):
                                    sim_list.append({'pik_id':ranked_similar_items[random_topk_rec_index][0], 'similarity':ranked_similar_items[random_topk_rec_index][1]})
                                
                            if len(sim_list) == 10:
                                    break

    
    ## 만약 sim_list가비어있거나 None이면, 그보다 더유사도가 적지만 그래도괜찮을수있는것을 추천해준다. 
    if sim_list == None:
        print('There is no recommended piks for your pik for now')
    
    elif bool(sim_list):
        return  sim_list #sorted(sim_dict, key=lambda x: x[1], reverse=True)    
    
    # elif not bool(sim_list):  
    elif not bool(sim_list): ##sim_list에 아무것도 존재하지 않으면, second_threshold를 사용해서 더낮지만 그래도 차선인 추천을 해준다. 
        print('There is no recommended piks for your pik for now')
                                    
    else:
        print('Hey, there are really not suitable recommendation for your pik for now. But we are working on it!')
                    



def get_most_similar_piks_en(pik_id, user_id, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold):
    
    #if pik_id not in num_link_by_pik.keys(): ##만약 픽에 링크가 하나도 없다면
    #    if pik_lang_dict[pik_id] == 'en':
    #        sim_list = list()
    #        while True: 
    #            key, value = choice(list(num_link_by_pik.items()))
    #            if pik_lang_dict[key] == 'en':
    #                if value > 10: ##10픽 추천해주기때문에
    #                    if pik_id != key and key not in user_pik[user_id] and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
    #                        sim_list.append({'user_id': key, 'similarity': 1})
    #                        if len(sim_list) == 10:
    #                            break


    #else:    
        
    sim = list()
        
    for uid, vec in piks_vec.items():
        # if pd.unique(data['language_code'][data['pik_id'] == int(pik_id)])[0] == 'en':
        if pik_lang_dict[uid] == 'en':
        
            thisSim = cosine_similarity(np.array(vec).reshape(1, -1), np.array(piks_vec[pik_id]).reshape(1, -1))
            sim.append((uid, thisSim[0][0]))

    full_ranked_similar_items = sorted(sim, key=lambda x: x[1], reverse=True) ##full similarity list
    ranked_similar_items = full_ranked_similar_items[:topk+1] ##only topk similarity list. 본픽도 들어가있기떄문에+1을해준다
    
    
    # sim_dict = {} ##추천 candidate 추려내서 저장하는 딕셔너리 
    sim_list = [] 
    for i in range(0, topk+1):
        if ranked_similar_items[i][1] > threshold:   
            if int(pik_id) != int(ranked_similar_items[i][0]) and int(ranked_similar_items[i][0]) not in user_pik[user_id] and ranked_similar_items[i][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                if i >= 1:  
                    if ranked_similar_items[i-1][1] != ranked_similar_items[i][1]: ##유사도가 바로그다음으로높은것과비교했을때 현재유사도와같으면 같은내용의픽이나링크일테니 그건스킵하라는것
                        if num_link_by_pik[ranked_similar_items[i][0]] >= num_link_threshold: ##픽안에 num_link_threshold 갯수이상 링크가 존재할때만 추천한다 
                            lottery = random()
                            if lottery <= 0.7:
                                sim_list.append({'pik_id':ranked_similar_items[i][0], 'similarity':ranked_similar_items[i][1]})

                                
                            elif 0.7 < lottery <= 0.88:
                                pik_title_sim = cosine_similarity(np.array(piktitle_vec[pik_id]).reshape(1, -1), np.array(piktitle_vec[ranked_similar_items[i][0]]).reshape(1, -1))[0][0]
                                if pik_title_sim >= piktitle_threshold: ##픽타이틀의 유사도도 threshold를 넘으면 그대로바로 추천리스트에들어간다 
                                    sim_list.append({'pik_id':ranked_similar_items[i][0], 'similarity':ranked_similar_items[i][1]})    

                            
                            elif 0.88 < lottery <= 1.0:
    
                                random_topk_rec_index = randint(0, topk)
                                if int(pik_id) != int(ranked_similar_items[random_topk_rec_index][0]) and int(ranked_similar_items[random_topk_rec_index][0]) not in user_pik[user_id] and ranked_similar_items[random_topk_rec_index][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]):
                                    sim_list.append({'pik_id':ranked_similar_items[random_topk_rec_index][0], 'similarity':ranked_similar_items[random_topk_rec_index][1]})
                                
                            if len(sim_list) == 10:
                                    break




    ## 만약 sim_list가비어있거나 None이면, 그보다 더유사도가 적지만 그래도괜찮을수있는것을 추천해준다. 
    if sim_list == None:
        print('There is no recommended piks for your pik for now')
    
    elif bool(sim_list):
        return  sim_list #sorted(sim_dict, key=lambda x: x[1], reverse=True)    

    # elif not bool(sim_list):  
    elif not bool(sim_list): ##sim_list에 아무것도 존재하지 않으면, second_threshold를 사용해서 더낮지만 그래도 차선인 추천을 해준다. 
        print('There is no recommended piks for your pik for now')
                               
    else:
        print('Hey, there are really not suitable recommendation for your pik for now. But we are working on it!')






def get_most_similar_piks_ko(pik_id, user_id, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold):
    
    
    ##만약 픽에 링크가 하나도 없다면
    #if pik_id not in num_link_by_pik.keys():
    #    if pik_lang_dict[pik_id] == 'ko':
    #        sim_list = list()
    #        while True: 
    #            key, value = choice(list(num_link_by_pik.items()))
    #            if pik_lang_dict[key] == 'ko':
    #                if value > 10: ##10픽 추천해주기때문에
    #                    if pik_id != key and key not in user_pik[user_id] and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
    #                        sim_list.append({'user_id': key, 'similarity': 1})
    #                        if len(sim_list) == 10:
    #                            break


    #else:

    sim = list()
    
    for uid, vec in piks_vec.items():
        # if pd.unique(data['language_code'][data['pik_id'] == int(pik_id)])[0] == 'ko':
        if pik_lang_dict[uid] == 'ko':
            thisSim = cosine_similarity(np.array(vec).reshape(1, -1), np.array(piks_vec[pik_id]).reshape(1, -1))
            sim.append((uid, thisSim[0][0]))

    full_ranked_similar_items = sorted(sim, key=lambda x: x[1], reverse=True) ##full similarity list
    ranked_similar_items = full_ranked_similar_items[:topk+1] ##only topk similarity list. 본픽도 들어가있기떄문에+1을해준다
    
    
    
    # sim_dict = {} ##추천 candidate 추려내서 저장하는 딕셔너리 
    sim_list = [] 
    for i in range(0, topk+1):
        if ranked_similar_items[i][1] > threshold:   
            if int(pik_id) != int(ranked_similar_items[i][0]) and int(ranked_similar_items[i][0]) not in user_pik[user_id] and ranked_similar_items[i][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                if i >= 1:  
                    if ranked_similar_items[i-1][1] != ranked_similar_items[i][1]: ##유사도가 바로그다음으로높은것과비교했을때 현재유사도와같으면 같은내용의픽이나링크일테니 그건스킵하라는것
                        if num_link_by_pik[ranked_similar_items[i][0]] >= num_link_threshold: ##픽안에 num_link_threshold 갯수이상 링크가 존재할때만 추천한다 
                            lottery = random()
                            if lottery <= 0.7:
                                sim_list.append({'pik_id':ranked_similar_items[i][0], 'similarity':ranked_similar_items[i][1]})

                                
                            elif 0.7 < lottery <= 0.88:
                                pik_title_sim = cosine_similarity(np.array(piktitle_vec[pik_id]).reshape(1, -1), np.array(piktitle_vec[ranked_similar_items[i][0]]).reshape(1, -1))[0][0]
                                if pik_title_sim >= piktitle_threshold: ##픽타이틀의 유사도도 threshold를 넘으면 그대로바로 추천리스트에들어간다 
                                    sim_list.append({'pik_id':ranked_similar_items[i][0], 'similarity':ranked_similar_items[i][1]})    

                            
                            elif 0.88 < lottery <= 1.0:
    
                                random_topk_rec_index = randint(0, topk)
                                if int(pik_id) != int(ranked_similar_items[random_topk_rec_index][0]) and int(ranked_similar_items[random_topk_rec_index][0]) not in user_pik[user_id] and ranked_similar_items[random_topk_rec_index][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]):
                                    sim_list.append({'pik_id':ranked_similar_items[random_topk_rec_index][0], 'similarity':ranked_similar_items[random_topk_rec_index][1]})
                                
                            if len(sim_list) == 10:
                                    break
        
    
    ## 만약 sim_list가비어있거나 None이면, 그보다 더유사도가 적지만 그래도괜찮을수있는것을 추천해준다. 
    if sim_list == None:
        print('안타깝게도 현재 최적의 픽 추천이 어려운 상황이네요 ㅠㅠ')
    
    elif bool(sim_list):
        return  sim_list #sorted(sim_dict, key=lambda x: x[1], reverse=True)    

    # elif not bool(sim_list):  
    elif not bool(sim_list): ##sim_list에 아무것도 존재하지 않으면, second_threshold를 사용해서 더낮지만 그래도 차선인 추천을 해준다. 
                 
       print('안타깝게도 현재 최적의 픽 추천이 어려운 상황이네요 ㅠㅠ')
                                    
    else:
        print('정말 죄송하지만 현재 최적의 픽 추천이 어려운 상황입니다 ㅠㅠ! 열일하고 있으니 조금만 기다려 주세요!')
              

def rec_pik_by_lang(pik_id, user_id, user_lang_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold):    
    # if pd.unique(data['language_code'][data['user_id'] == int(user_id)])[0] == 'ko': ##language_cde가 'ko' 인지, 'en'인지,
    if user_lang_dict[user_id] == 'ko':
        result = get_most_similar_piks_ko(pik_id, user_id, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
        return result
    # elif pd.unique(data['language_code'][data['user_id'] == int(user_id)])[0] == 'en': ##language_cde가 'ko' 인지, 'en'인지,
    elif user_lang_dict[user_id] == 'en':
        result = get_most_similar_piks_en(pik_id, user_id, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
        return result
    else: 
        result = get_most_similar_piks(pik_id, user_id, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
        return result

## This is modification of above to inlcude language condition
input_spec = Multipart(user_id=Text(), pik_id=Text())
@svc.api(input=input_spec, output=JSON())
def predict(user_id, pik_id) -> dict:
    
    similarity_dict = rec_pik_by_lang(pik_id, user_id, user_lang_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik,  topk=40, threshold=0.7, piktitle_threshold=0.77, num_link_threshold=3)
    return similarity_dict #sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)




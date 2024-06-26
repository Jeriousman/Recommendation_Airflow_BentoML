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
from collections import Counter
import fasttext




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

with open("/opt/airflow/dags/data/user_lang_dict_userset.json") as f:
    user_lang_dict_userset = json.load(f)
 
with open("/opt/airflow/dags/data/pik_lang_dict_userset.json") as f:
    pik_lang_dict_userset = json.load(f)

with open("/opt/airflow/dags/data/user_lang_dict_detected.json") as f:
    user_lang_dict_detected = json.load(f)
 
with open("/opt/airflow/dags/data/pik_lang_dict_detected.json") as f:
    pik_lang_dict_detected = json.load(f)
    
with open('/opt/airflow/dags/data/pik_link.json') as f:
    pik_link = json.load(f)

with open('/opt/airflow/dags/data/user_pik.json') as f:
    user_pik = json.load(f)
    
with open('/opt/airflow/dags/data/user_link.json') as f:
    user_link = json.load(f)
    
with open('/opt/airflow/dags/data/pik_status_dict.json') as f:
    pik_status_dict = json.load(f)


with open('/opt/airflow/dags/data/linkid_title_dict.json') as f:
    linkid_title_dict = json.load(f)

with open('/opt/airflow/dags/data/pikid_title_dict.json') as f:
    pikid_title_dict = json.load(f)

# path_to_pretrained_model = '/home/hojun/Downloads/lid.176.bin'
# fmodel = fasttext.load_model(path_to_pretrained_model)


















def get_most_similar_piks(pik_id, user_id, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold):
    sim = list()
        
    for uid, vec in piks_vec.items():

        thisSim = cosine_similarity(np.array(vec).reshape(1, -1), np.array(piks_vec[pik_id]).reshape(1, -1))
        sim.append((uid, thisSim[0][0]))
    
    not_thispik_not_mypik_rec = [] ## filtering
    for (pid, similarity) in sim:
        if int(pik_id) != pid and int(pid) not in user_pik[user_id]: ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
            not_thispik_not_mypik_rec.append((pid, similarity))
            
    if not not_thispik_not_mypik_rec: ##if not_me_not_friends_rec is empty list,
        # pass
        return not_thispik_not_mypik_rec
    else:
        full_ranked_similar_items = sorted(not_thispik_not_mypik_rec, key=lambda x: x[1], reverse=True) ##full similarity list
        ranked_similar_items = full_ranked_similar_items[:topk+1] ##only topk similarity list. 본픽도 들어가있기떄문에+1을해준다

            
        # sim_dict = {} ##추천 candidate 추려내서 저장하는 딕셔너리 
        sim_list = [] 
        for i in range(0, topk+1):
            if ranked_similar_items[i][1] > threshold:   
                #if int(pik_id) != int(ranked_similar_items[i][0]) and int(ranked_similar_items[i][0]) not in user_pik[user_id] and ranked_similar_items[i][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                if ranked_similar_items[i][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
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
            return sim_list
                                        
        else:
            print('Hey, there are really not suitable recommendation for your pik for now. But we are working on it!')
                        




def get_most_similar_piks_en(pik_id, user_id, pik_lang_dict, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold):
    
 
        
    sim = list()
        
    for pid, vec in piks_vec.items():
        if status_dict[pid] == 'public':# if pd.unique(data['language_code'][data['pik_id'] == int(pik_id)])[0] == 'en':
            if pik_lang_dict[pid] == 'en':
        
                thisSim = cosine_similarity(np.array(vec).reshape(1, -1), np.array(piks_vec[pik_id]).reshape(1, -1))
                sim.append((pid, thisSim[0][0]))

    not_thispik_not_mypik_rec = [] ## filtering
    for (pid, similarity) in sim:
        try:
            if int(pik_id) != pid and int(pid) not in user_pik[user_id]: ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                not_thispik_not_mypik_rec.append((pid, similarity))
        except KeyError:  ##유저가 링크/픽이 하나도 없더라도 추천 받을 수 있도록 하기 위함이다. 
            not_thispik_not_mypik_rec.append((pid, similarity))
                
    if not not_thispik_not_mypik_rec: ##if not_me_not_friends_rec is empty list,
        # pass
        return not_thispik_not_mypik_rec
    else:
        full_ranked_similar_items = sorted(not_thispik_not_mypik_rec, key=lambda x: x[1], reverse=True) ##full similarity list
        ranked_similar_items = full_ranked_similar_items[:topk+1] ##only topk similarity list. 본픽도 들어가있기떄문에+1을해준다
    
        
        # sim_dict = {} ##추천 candidate 추려내서 저장하는 딕셔너리 
        sim_list = [] 
        for i in range(0, topk+1):
            if ranked_similar_items[i][1] > threshold:   
                #if int(pik_id) != int(ranked_similar_items[i][0]) and int(ranked_similar_items[i][0]) not in user_pik[user_id] and ranked_similar_items[i][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                if ranked_similar_items[i][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
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
                                    try:
                                        if int(pik_id) != int(ranked_similar_items[random_topk_rec_index][0]) and int(ranked_similar_items[random_topk_rec_index][0]) not in user_pik[user_id] and ranked_similar_items[random_topk_rec_index][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]):
                                            sim_list.append({'pik_id':ranked_similar_items[random_topk_rec_index][0], 'similarity':ranked_similar_items[random_topk_rec_index][1]})
                                    except KeyError: ##링크나 픽이 하나도 없는 유저도 다른 픽에서 추천을 받기 위해서
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
            return sim_list
                                
        else:
            print('Hey, there are really not suitable recommendation for your pik for now. But we are working on it!')










def get_most_similar_piks_ko(pik_id, user_id, pik_lang_dict, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold):
    


    sim = list()


    
    for pid, vec in piks_vec.items():
        if status_dict[pid] == 'public':
            if pik_lang_dict[pid] == 'ko' or pik_lang_dict[pid] == 'kr':
                thisSim = cosine_similarity(np.array(vec).reshape(1, -1), np.array(piks_vec[pik_id]).reshape(1, -1))
                sim.append((pid, thisSim[0][0]))


    not_thispik_not_mypik_rec = [] ## filtering
    for (pid, similarity) in sim:
        try:
            if int(pik_id) != pid and int(pid) not in user_pik[user_id]: ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                not_thispik_not_mypik_rec.append((pid, similarity))
                
        except KeyError:  ##유저가 링크/픽이 하나도 없더라도 추천 받을 수 있도록 하기 위함이다. 
            not_thispik_not_mypik_rec.append((pid, similarity))
            
            
    if not not_thispik_not_mypik_rec: ##if not_me_not_friends_rec is empty list,
        # pass
        return not_thispik_not_mypik_rec
    else:
        full_ranked_similar_items = sorted(not_thispik_not_mypik_rec, key=lambda x: x[1], reverse=True) ##full similarity list
        ranked_similar_items = full_ranked_similar_items[:topk+1] ##only topk similarity list. 본픽도 들어가있기떄문에+1을해준다
        
        
        
        # sim_dict = {} ##추천 candidate 추려내서 저장하는 딕셔너리 
        sim_list = [] 
        for i in range(0, topk+1):
            if ranked_similar_items[i][1] > threshold:   
                #if int(pik_id) != int(ranked_similar_items[i][0]) and int(ranked_similar_items[i][0]) not in user_pik[user_id] and ranked_similar_items[i][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                if ranked_similar_items[i][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것    
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
                                    try:
                                        if int(pik_id) != int(ranked_similar_items[random_topk_rec_index][0]) and int(ranked_similar_items[random_topk_rec_index][0]) not in user_pik[user_id] and ranked_similar_items[random_topk_rec_index][0] not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]):
                                            sim_list.append({'pik_id':ranked_similar_items[random_topk_rec_index][0], 'similarity':ranked_similar_items[random_topk_rec_index][1]})
                                    except KeyError: ##링크나 픽이 하나도 없는 유저도 다른 픽에서 추천을 받기 위해서
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
            return sim_list
                                        
        else:
            print('정말 죄송하지만 현재 최적의 픽 추천이 어려운 상황입니다 ㅠㅠ! 열일하고 있으니 조금만 기다려 주세요!')




# def rec_pik_by_lang_fasttext(pik_id, user_id, status_dict, user_lang_dict, pik_lang_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold):    

#     ##predict user language
#     lang_pred_user = [fmodel.predict([linkid_title_dict[str(link_id)]])[0][0][0][-2:] for link_id in user_link[user_id]]
#     language_pred_count_user_dict = Counter(lang_pred_user)
#     final_pred_lang_user = [k for k, v in language_pred_count_user_dict.items() if v == max(language_pred_count_user_dict.values())][0]

#     ##predict pik language
#     lang_pred_pik = [fmodel.predict([linkid_title_dict[str(link_id)]])[0][0][0][-2:] for link_id in pik_link[pik_id]]
#     language_pred_count_pik_dict = Counter(lang_pred_pik)
#     final_pred_lang_pik = [k for k, v in language_pred_count_pik_dict.items() if v == max(language_pred_count_pik_dict.values())][0]
    
    
    
#     if final_pred_lang_user == 'ko' or final_pred_lang_user == 'kr':
        
#         if final_pred_lang_pik == 'ko' or final_pred_lang_pik == 'kr':
#         # if final_pred_lang_pik == 'en':
#             if pik_id in num_link_by_pik.keys():
#                 result = get_most_similar_piks_ko(pik_id, user_id, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk=10, threshold=0.3, piktitle_threshold=0.3, num_link_threshold=3)
#                 return result
            
#             elif pik_id not in num_link_by_pik.keys():
#                 print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
#                 sim_list = list()
#                 while True: 
#                     key, value = choice(list(num_link_by_pik.items()))
#                     if status_dict[key] == 'public':
#                         if pik_lang_dict[key] == 'ko' or pik_lang_dict[key] == 'kr':
#                             if value > 10: ##10픽 이상인 것을 추천해주기때문에
#                                 if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                
#                                     sim_list.append({'pik_id': key, 'similarity': 1})
#                                     if len(sim_list) == 10:
#                                         break    
#             return sim_list

#     ##만약 유저는 한국어로 설정되어있지만 픽추천을 받고 싶은 픽은 영어로 설정되어있다면 유저의 언어를 따라서 추천해줘라.

#         elif final_pred_lang_pik == 'en' or final_pred_lang_pik != 'ko' or final_pred_lang_pik != 'kr':
#             if pik_id in num_link_by_pik.keys():
#                 result = get_most_similar_piks_ko(pik_id, user_id, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
#                 return result
            
#             elif pik_id not in num_link_by_pik.keys():
#                 print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
#                 sim_list = list()
#                 while True: 
#                     key, value = choice(list(num_link_by_pik.items()))
#                     if status_dict[key] == 'public':
#                         if pik_lang_dict[key] == 'ko' or pik_lang_dict[key] == 'kr':
#                             if value > 10: ##10픽 이상인 것을 추천해주기때문에
#                                 if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                
#                                     sim_list.append({'pik_id': key, 'similarity': 1})
#                                     if len(sim_list) == 10:
#                                         break    
#             return sim_list



        
#     elif final_pred_lang_user == 'en':
        
#         if final_pred_lang_pik == 'en' or final_pred_lang_pik != 'ko' or final_pred_lang_pik != 'kr':
#             if pik_id in num_link_by_pik.keys():
#                 result = get_most_similar_piks_en(pik_id, user_id, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
#                 return result
            
#             elif pik_id not in num_link_by_pik.keys():
#                 print('유저는 영어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
#                 sim_list = list()
#                 while True: 
#                     key, value = choice(list(num_link_by_pik.items()))
#                     if status_dict[key] == 'public':
#                         if pik_lang_dict[key] == 'en':
#                             if value > 10: ##10픽 이상인 것을 추천해주기때문에
#                                 if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                
#                                     sim_list.append({'pik_id': key, 'similarity': 1})
#                                     if len(sim_list) == 10:
#                                         break  
                
#             return sim_list
    
        
    

  

#     ##만약 유저는 영어로 설정되어있지만 픽추천을 받고 싶은 픽은 한국어로 설정되어있다면 유저의 언어를 따라서 추천해줘라.

#         elif final_pred_lang_pik == 'ko' or final_pred_lang_pik == 'kr':
#             if pik_id in num_link_by_pik.keys(): ##그 픽 아이디가 우리 픽 풀에 존재한다면,
#                 result = get_most_similar_piks_en(pik_id, user_id, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
#                 return result
            
#             elif pik_id not in num_link_by_pik.keys():
#                 print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
#                 sim_list = list()
#                 while True: 
#                     key, value = choice(list(num_link_by_pik.items()))
#                     if status_dict[key] == 'public':
#                         if pik_lang_dict[key] == 'en':
#                             if value > 10: ##10픽 이상인 것을 추천해주기때문에
#                                 if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                
#                                     sim_list.append({'pik_id': key, 'similarity': 1})
#                                     if len(sim_list) == 10:
#                                         break    
#             return sim_list


#     elif final_pred_lang_user != 'ko' or final_pred_lang_user != 'kr' or final_pred_lang_user != 'en':
#         final_pred_lang_user = 'en'
        
#         if final_pred_lang_pik == 'en':
#             if pik_id in num_link_by_pik.keys():
#                 result = get_most_similar_piks_en(pik_id, user_id, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
#                 return result
            
#             elif pik_id not in num_link_by_pik.keys():
#                 print('유저는 영어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
#                 sim_list = list()
#                 while True: 
#                     key, value = choice(list(num_link_by_pik.items()))
#                     if status_dict[key] == 'public':
#                         if pik_lang_dict[key] == 'en':
#                             if value > 10: ##10픽 이상인 것을 추천해주기때문에
#                                 if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                
#                                     sim_list.append({'pik_id': key, 'similarity': 1})
#                                     if len(sim_list) == 10:
#                                         break  
                
#             return sim_list



#         elif final_pred_lang_pik == 'ko' or final_pred_lang_pik == 'kr' or final_pred_lang_pik != 'en':
#             if pik_id in num_link_by_pik.keys():
#                 result = get_most_similar_piks_en(pik_id, user_id, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
#                 return result
            
#             elif pik_id not in num_link_by_pik.keys():
#                 print('유저는 영어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
#                 sim_list = list()
#                 while True: 
#                     key, value = choice(list(num_link_by_pik.items()))
#                     if status_dict[key] == 'public':
#                         if pik_lang_dict[key] == 'en':
#                             if value > 10: ##10픽 이상인 것을 추천해주기때문에
#                                 if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                
#                                     sim_list.append({'pik_id': key, 'similarity': 1})
#                                     if len(sim_list) == 10:
#                                         break  
                
#             return sim_list




        

#     elif user_id not in user_lang_dict.keys():   ##만약 유저가 아직 업데이트 안된 신규 유저라면
#         if pik_id not in num_link_by_pik.keys(): ##만약 픽에 링크가 하나도 없다면
#             print('유저언어는 관심없고 유저등록도 안되었고 픽 등록도 안되어서 암거나 추천해준다')
#             sim_list = list()
#             while True: 
#                 key, value = choice(list(num_link_by_pik.items()))
#                 if status_dict[key] == 'public':
#                     if value > 10: ##10픽 이상인 것을 추천해주기때문에
#                         if pik_id != key and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                        
#                             sim_list.append({'pik_id': key, 'similarity': 1})
#                             if len(sim_list) == 10:
#                                 break
#             return sim_list





# piks_vec 
# piktitle_vec
# num_link_by_pik
# user_lang_dict_userset
# pik_lang_dict_userset
# user_lang_dict_detected
# pik_lang_dict_detected
# pik_link 
# user_pik 
# user_link 
# pik_status_dict 
# linkid_title_dict
# pikid_title_dict
# status_dict = pik_status_dict

# topk=10
# threshold=0.6
# piktitle_threshold=0.6
# num_link_threshold=2
# min_user_link_num=3

# user_id = '3412'
# pik_id = '17049'


# num_link_by_pik
def rec_pik_by_lang(pik_id, user_id, status_dict, user_lang_dict_detected, user_lang_dict_userset, pik_lang_dict_detected, pik_lang_dict_userset, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold, min_user_link_num):    
    # try:
        # if pd.unique(data['language_code'][data['user_id'] == int(user_id)])[0] == 'ko': ##language_cde가 'ko' 인지, 'en'인지,
    if user_id in user_link.keys(): ##등록되어있는 유저들 중에,
        
        if len(user_link[user_id]) >= min_user_link_num: 
            if (user_lang_dict_detected[user_id] == 'ko' or user_lang_dict_detected[user_id] == 'kr') and (pik_lang_dict_detected[pik_id] == 'ko' or pik_lang_dict_detected[pik_id] == 'kr'):
                '''
                유저의 링크들의 합이 가장 많은 수가 한국어 일 때 추천해 는 로직 
                '''
            # if (user_lang_dict_detected[user_id] == 'ko' or user_lang_dict_detected[user_id] == 'kr') and (pik_lang_dict_detected[pik_id] == 'ko' or pik_lang_dict[pik_id] == 'kr'):
                if pik_id in num_link_by_pik.keys():
                    '''
                    현 픽이 트레이닝 시에 존재 했다면
                    '''
                    result = get_most_similar_piks_ko(pik_id, user_id, pik_lang_dict_detected, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
                    return result
                
                elif pik_id not in num_link_by_pik.keys():
                    print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
                    sim_list = list()
                    while True: 
                        key, value = choice(list(num_link_by_pik.items()))
                        if status_dict[key] == 'public':
                            if pik_lang_dict_detected[key] == 'ko' or pik_lang_dict_detected[key] == 'kr':
                                if value > 10: ##10픽 이상인 것을 추천해주기때문에
                                    if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                    
                                        sim_list.append({'pik_id': key, 'similarity': 1})
                                        if len(sim_list) == 10:
                                            break    
                return sim_list
            
            
            elif user_lang_dict_detected[user_id] == 'en' and pik_lang_dict_detected[pik_id] == 'en':
                '''
                유저가 n개보다 링크가 적고 영어
                '''
                if pik_id in num_link_by_pik.keys():
                    result = get_most_similar_piks_en(pik_id, user_id, pik_lang_dict_detected, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
                    return result
                
                elif pik_id not in num_link_by_pik.keys():
                    print('유저는 영어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
                    sim_list = list()
                    while True: 
                        key, value = choice(list(num_link_by_pik.items()))
                        if status_dict[key] == 'public':
                            if pik_lang_dict_detected[key] == 'ko' or pik_lang_dict_detected[key] == 'kr':
                                if value > 10: ##10픽 이상인 것을 추천해주기때문에
                                    if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                        sim_list.append({'pik_id': key, 'similarity': 1})
                                        if len(sim_list) == 10:
                                            break  
                    
                return sim_list
            
            
            ##만약 유저는 한국어로 설정되어있지만 픽추천을 받고 싶은 픽은 영어로 설정되어있다면 유저의 언어를 따라서 추천해줘라.
            elif (user_lang_dict_detected[user_id] == 'ko' or user_lang_dict_detected[user_id] == 'kr') and pik_lang_dict_detected[pik_id] == 'en': 
    
                if pik_id in num_link_by_pik.keys():
                    result = get_most_similar_piks_ko(pik_id, user_id, pik_lang_dict_detected, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
                    return result
                
                elif pik_id not in num_link_by_pik.keys():
                    print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
                    sim_list = list()
                    while True: 
                        key, value = choice(list(num_link_by_pik.items()))
                        if status_dict[key] == 'public':
                            if pik_lang_dict_detected[key] == 'ko' or pik_lang_dict_detected[key] == 'kr':
                                if value > 10: ##10픽 이상인 것을 추천해주기때문에
                                    if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                    
                                        sim_list.append({'pik_id': key, 'similarity': 1})
                                        if len(sim_list) == 10:
                                            break    
                return sim_list
            

            ##만약 유저는 영어로 설정되어있지만 픽추천을 받고 싶은 픽은 한국어로 설정되어있다면 유저의 언어를 따라서 추천해줘라.
            elif user_lang_dict_detected[user_id] == 'en' and (pik_lang_dict_detected[pik_id] == 'ko' or pik_lang_dict_detected[pik_id] == 'kr'):
                if pik_id in num_link_by_pik.keys(): ##그 픽 아이디가 우리 픽 풀에 존재한다면,
                    result = get_most_similar_piks_en(pik_id, user_id, pik_lang_dict_detected, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
                    return result
                
                elif pik_id not in num_link_by_pik.keys():
                    print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
                    sim_list = list()
                    while True: 
                        key, value = choice(list(num_link_by_pik.items()))
                        if status_dict[key] == 'public':
                            if pik_lang_dict_detected[key] == 'en':
                                if value > 10: ##10픽 이상인 것을 추천해주기때문에
                                    if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                    
                                        sim_list.append({'pik_id': key, 'similarity': 1})
                                        if len(sim_list) == 10:
                                            break    
                return sim_list
            

           
            elif user_lang_dict_detected[user_id] != 'en' or user_lang_dict_detected[user_id] != 'ko' or user_lang_dict_detected[user_id] != 'kr':
                '''
                detected 된 언어가 모두 영어나 한국어가 아니라면 영어로 간주하고 추천해라 
                '''
                # if pik_lang_dict[pik_id] == 'ko' or pik_lang_dict[pik_id] == 'kr':
                if pik_id in num_link_by_pik.keys(): ##그 픽 아이디가 우리 픽 풀에 존재한다면,
                    result = get_most_similar_piks_en(pik_id, user_id, pik_lang_dict_detected, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
                    return result
                
                elif pik_id not in num_link_by_pik.keys():
                    print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
                    sim_list = list()
                    while True: 
                        key, value = choice(list(num_link_by_pik.items()))
                        if status_dict[key] == 'public':
                            if pik_lang_dict_detected[key] == 'en':
                                if value > 10: ##10픽 이상인 것을 추천해주기때문에
                                    if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                    
                                        sim_list.append({'pik_id': key, 'similarity': 1})
                                        if len(sim_list) == 10:
                                            break    
                return sim_list
                
            
            



        
        
        elif len(user_link[user_id]) < min_user_link_num:
            if (user_lang_dict_userset[user_id] == 'ko' or user_lang_dict_userset[user_id] == 'kr') and (pik_lang_dict_userset[pik_id] == 'ko' or pik_lang_dict_userset[pik_id] == 'kr'):
                '''
                유저가 한국어로 주언어를 셋팅 해 놓았을 때 추천하는 법
                '''
                if pik_id in num_link_by_pik.keys():
                    result = get_most_similar_piks_ko(pik_id, user_id, pik_lang_dict_userset, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
                    return result
                
                elif pik_id not in num_link_by_pik.keys():
                    print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
                    sim_list = list()
                    while True: 
                        key, value = choice(list(num_link_by_pik.items()))
                        if status_dict[key] == 'public':
                            if pik_lang_dict_userset[key] == 'ko' or pik_lang_dict_userset[key] == 'kr':
                                if value > 10: ##10픽 이상인 것을 추천해주기때문에
                                    if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                    
                                        sim_list.append({'pik_id': key, 'similarity': 1})
                                        if len(sim_list) == 10:
                                            break    
                return sim_list      
            
            
            
            elif user_lang_dict_userset[user_id] == 'en' and pik_lang_dict_userset[pik_id] == 'en':
                '''
                유저가 n개보다 링크가 적고 영어를 주언어로 해 놓았을 때는
                '''
                if pik_id in num_link_by_pik.keys():
                    result = get_most_similar_piks_en(pik_id, user_id, pik_lang_dict_userset, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
                    return result
                
                elif pik_id not in num_link_by_pik.keys():
                    print('유저는 영어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
                    sim_list = list()
                    while True: 
                        key, value = choice(list(num_link_by_pik.items()))
                        if status_dict[key] == 'public':
                            if pik_lang_dict_userset[key] == 'ko' or pik_lang_dict_userset[key] == 'kr':
                                if value > 10: ##10픽 이상인 것을 추천해주기때문에
                                    if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                        sim_list.append({'pik_id': key, 'similarity': 1})
                                        if len(sim_list) == 10:
                                            break  
                    
                return sim_list
    
    


            ##만약 유저는 한국어로 설정되어있지만 픽추천을 받고 싶은 픽은 영어로 설정되어있다면 유저의 언어를 따라서 추천해줘라.
            elif (user_lang_dict_userset[user_id] == 'ko' or user_lang_dict_userset[user_id] == 'kr') and pik_lang_dict_userset[pik_id] == 'en': 
    
                if pik_id in num_link_by_pik.keys():
                    result = get_most_similar_piks_ko(pik_id, user_id, pik_lang_dict_userset, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
                    return result
                
                elif pik_id not in num_link_by_pik.keys():
                    print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
                    sim_list = list()
                    while True: 
                        key, value = choice(list(num_link_by_pik.items()))
                        if status_dict[key] == 'public':
                            if pik_lang_dict_userset[key] == 'ko' or pik_lang_dict_userset[key] == 'kr':
                                if value > 10: ##10픽 이상인 것을 추천해주기때문에
                                    if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                    
                                        sim_list.append({'pik_id': key, 'similarity': 1})
                                        if len(sim_list) == 10:
                                            break    
                return sim_list        
        




            ##만약 유저는 영어로 설정되어있지만 픽추천을 받고 싶은 픽은 한국어로 설정되어있다면 유저의 언어를 따라서 추천해줘라.
            elif user_lang_dict_userset[user_id] == 'en' and (pik_lang_dict_userset[pik_id] == 'ko' or pik_lang_dict_userset[pik_id] == 'kr'):
                if pik_id in num_link_by_pik.keys(): ##그 픽 아이디가 우리 픽 풀에 존재한다면,
                    result = get_most_similar_piks_en(pik_id, user_id, pik_lang_dict_userset, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
                    return result
                
                elif pik_id not in num_link_by_pik.keys():
                    print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
                    sim_list = list()
                    while True: 
                        key, value = choice(list(num_link_by_pik.items()))
                        if status_dict[key] == 'public':
                            if pik_lang_dict_userset[key] == 'en':
                                if value > 10: ##10픽 이상인 것을 추천해주기때문에
                                    if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                    
                                        sim_list.append({'pik_id': key, 'similarity': 1})
                                        if len(sim_list) == 10:
                                            break    
                return sim_list


 
            ## 아래코드는 주언어에 auto가 생기면 수정해서 쓰면 된다
            
            # ##만약 유저는 영어로 설정되어있지만 픽추천을 받고 싶은 픽은 한국어로 설정되어있다면 유저의 언어를 따라서 추천해줘라.
            # elif user_lang_dict_userset[user_id] != 'en' or (user_lang_dict_userset[user_id] != 'ko' or user_lang_dict_userset[user_id] != 'kr'):
            #     # if pik_lang_dict[pik_id] == 'ko' or pik_lang_dict[pik_id] == 'kr':
            #     if pik_id in num_link_by_pik.keys(): ##그 픽 아이디가 우리 픽 풀에 존재한다면,
            #         result = get_most_similar_piks_en(pik_id, user_id, pik_lang_dict_userset, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
            #         return result
                
            #     elif pik_id not in num_link_by_pik.keys():
            #         print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
            #         sim_list = list()
            #         while True: 
            #             key, value = choice(list(num_link_by_pik.items()))
            #             if status_dict[key] == 'public':
            #                 if pik_lang_dict_userset[key] == 'en':
            #                     if value > 10: ##10픽 이상인 것을 추천해주기때문에
            #                         if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                    
            #                             sim_list.append({'pik_id': key, 'similarity': 1})
            #                             if len(sim_list) == 10:
            #                                 break    
            #     return sim_list


    if user_id not in user_link.keys(): ##유저가 링크가 하나도 없으면 
        if ((user_lang_dict_userset[user_id] == 'ko' or user_lang_dict_userset[user_id] == 'kr') and (pik_lang_dict_userset[pik_id] == 'ko' or pik_lang_dict_userset[pik_id] == 'kr')) or ((user_lang_dict_userset[user_id] == 'ko' or user_lang_dict_userset[user_id] == 'kr') and pik_lang_dict_userset[pik_id] == 'en' ):
            if pik_id in num_link_by_pik.keys():
                result = get_most_similar_piks_ko(pik_id, user_id, pik_lang_dict_userset, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
                # result = get_most_similar_piks_ko(pik_id, user_id, pik_lang_dict_userset, pik_status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk=5, threshold=0.7, piktitle_threshold=0.7, num_link_threshold=3)
                return result
            
            if pik_id not in num_link_by_pik.keys():
                print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
                sim_list = list()
                while True: 
                    key, value = choice(list(num_link_by_pik.items()))
                    if status_dict[key] == 'public':
                        if pik_lang_dict_userset[key] == 'ko' or pik_lang_dict_userset[key] == 'kr':
                            if value > 10: ##10픽 이상인 것을 추천해주기때문에
                                if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                
                                    sim_list.append({'pik_id': key, 'similarity': 1})
                                    if len(sim_list) == 10:
                                        break    
            return sim_list               


        elif (user_lang_dict_userset[user_id] == 'en' and (pik_lang_dict_userset[pik_id] == 'ko' or pik_lang_dict_userset[pik_id] == 'kr')) or (user_lang_dict_userset[user_id] == 'en' and pik_lang_dict_userset[pik_id] == 'en'):
            if pik_id in num_link_by_pik.keys():
                result = get_most_similar_piks_en(pik_id, user_id, pik_lang_dict_userset, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
                return result
            
            elif pik_id not in num_link_by_pik.keys():
                print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
                sim_list = list()
                while True: 
                    key, value = choice(list(num_link_by_pik.items()))
                    if status_dict[key] == 'public':
                        if pik_lang_dict_userset[key] == 'en' or pik_lang_dict_userset[key] == 'en':
                            if value > 10: ##10픽 이상인 것을 추천해주기때문에
                                if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                
                                    sim_list.append({'pik_id': key, 'similarity': 1})
                                    if len(sim_list) == 10:
                                        break    
            return sim_list   


    # except KeyError: ##만약에 유저가 링크가 하나도 없다면 리스트에 없기 때문에 key error가 뜰 것이다
    #     if ((user_lang_dict_userset[user_id] == 'ko' or user_lang_dict_userset[user_id] == 'kr') and (pik_lang_dict_userset[pik_id] == 'ko' or pik_lang_dict_userset[pik_id] == 'kr')) or ((user_lang_dict_userset[user_id] == 'ko' or user_lang_dict_userset[user_id] == 'kr') and pik_lang_dict_userset[pik_id] == 'en' ):
    #         if pik_id in num_link_by_pik.keys():
    #             result = get_most_similar_piks_ko(pik_id, user_id, pik_lang_dict_userset, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
    #             return result
            
    #         elif pik_id not in num_link_by_pik.keys():
    #             print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
    #             sim_list = list()
    #             while True: 
    #                 key, value = choice(list(num_link_by_pik.items()))
    #                 if status_dict[key] == 'public':
    #                     if pik_lang_dict_userset[key] == 'ko' or pik_lang_dict_userset[key] == 'kr':
    #                         if value > 10: ##10픽 이상인 것을 추천해주기때문에
    #                             if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                
    #                                 sim_list.append({'pik_id': key, 'similarity': 1})
    #                                 if len(sim_list) == 10:
    #                                     break    
    #         return sim_list     
        
        
        

            
# else: 
#     result = get_most_similar_piks(pik_id, user_id, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
#     return result
        

        
        
        
        
        
        
        
        
        





# def rec_pik_by_lang_original(pik_id, user_id, status_dict, user_lang_dict, pik_lang_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold):    
#     # if pd.unique(data['language_code'][data['user_id'] == int(user_id)])[0] == 'ko': ##language_cde가 'ko' 인지, 'en'인지,
#     if user_id in user_lang_dict.keys():
#         if user_lang_dict[user_id] == 'ko' or user_lang_dict[user_id] == 'kr':
#             if pik_lang_dict[pik_id] == 'ko' or pik_lang_dict[pik_id] == 'kr':
#                 if pik_id in num_link_by_pik.keys():
#                     result = get_most_similar_piks_ko(pik_id, user_id, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
                    
#                     return result
                
#                 elif pik_id not in num_link_by_pik.keys():
#                     print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
#                     sim_list = list()
#                     while True: 
#                         key, value = choice(list(num_link_by_pik.items()))
#                         if status_dict[key] == 'public':
#                             if pik_lang_dict[key] == 'ko' or pik_lang_dict[key] == 'kr':
#                                 if value > 10: ##10픽 이상인 것을 추천해주기때문에
#                                     if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                    
#                                         sim_list.append({'pik_id': key, 'similarity': 1})
#                                         if len(sim_list) == 10:
#                                             break    
#                 return sim_list
            


            
#         elif user_lang_dict[user_id] == 'en':
#             if pik_lang_dict[pik_id] == 'en':
#                 if pik_id in num_link_by_pik.keys():
#                     result = get_most_similar_piks_en(pik_id, user_id, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
#                     return result
                
#                 elif pik_id not in num_link_by_pik.keys():
#                     print('유저는 영어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
#                     sim_list = list()
#                     while True: 
#                         key, value = choice(list(num_link_by_pik.items()))
#                         if status_dict[key] == 'public':
#                             if pik_lang_dict[key] == 'ko' or pik_lang_dict[key] == 'kr':
#                                 if value > 10: ##10픽 이상인 것을 추천해주기때문에
#                                     if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                    
#                                         sim_list.append({'pik_id': key, 'similarity': 1})
#                                         if len(sim_list) == 10:
#                                             break  
                    
#                 return sim_list
        
            
        
#         ##만약 유저는 한국어로 설정되어있지만 픽추천을 받고 싶은 픽은 영어로 설정되어있다면 유저의 언어를 따라서 추천해줘라.
#         elif user_lang_dict[user_id] == 'ko' or user_lang_dict[user_id] == 'kr': 
#             if pik_lang_dict[pik_id] == 'en':
#                 if pik_id in num_link_by_pik.keys():
#                     result = get_most_similar_piks_ko(pik_id, user_id, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
#                     return result
                
#                 elif pik_id not in num_link_by_pik.keys():
#                     print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
#                     sim_list = list()
#                     while True: 
#                         key, value = choice(list(num_link_by_pik.items()))
#                         if status_dict[key] == 'public':
#                             if pik_lang_dict[key] == 'ko' or pik_lang_dict[key] == 'kr':
#                                 if value > 10: ##10픽 이상인 것을 추천해주기때문에
#                                     if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                    
#                                         sim_list.append({'pik_id': key, 'similarity': 1})
#                                         if len(sim_list) == 10:
#                                             break    
#                 return sim_list

      

#         ##만약 유저는 영어로 설정되어있지만 픽추천을 받고 싶은 픽은 한국어로 설정되어있다면 유저의 언어를 따라서 추천해줘라.
#         if user_lang_dict[user_id] == 'en':
#             if pik_lang_dict[pik_id] == 'ko' or pik_lang_dict[pik_id] == 'kr':
#                 if pik_id in num_link_by_pik.keys(): ##그 픽 아이디가 우리 픽 풀에 존재한다면,
#                     result = get_most_similar_piks_en(pik_id, user_id, status_dict, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
#                     return result
                
#                 elif pik_id not in num_link_by_pik.keys():
#                     print('유저는 한국어를 사용하고 업데이트 됐으나 픽은 링크가 없거나 업데이트가 안되었다. 그러므로 랜덤추천을 한다')
#                     sim_list = list()
#                     while True: 
#                         key, value = choice(list(num_link_by_pik.items()))
#                         if status_dict[key] == 'public':
#                             if pik_lang_dict[key] == 'en':
#                                 if value > 10: ##10픽 이상인 것을 추천해주기때문에
#                                     if pik_id != key and pik_id not in pik_link.keys() and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                                    
#                                         sim_list.append({'pik_id': key, 'similarity': 1})
#                                         if len(sim_list) == 10:
#                                             break    
#                 return sim_list

            
    
#     elif user_id not in user_lang_dict.keys():   ##만약 유저가 아직 업데이트 안된 신규 유저라면
#         if pik_id not in num_link_by_pik.keys(): ##만약 픽에 링크가 하나도 없다면
#             print('유저언어는 관심없고 유저등록도 안되었고 픽 등록도 안되어서 암거나 추천해준다')
#             sim_list = list()
#             while True: 
#                 key, value = choice(list(num_link_by_pik.items()))
#                 if status_dict[key] == 'public':
#                     if value > 10: ##10픽 이상인 것을 추천해주기때문에
#                         if pik_id != key and key not in list([sim_list[num]['pik_id'] for num in range(len(sim_list))]): ##본픽이 아니고 현 추천픽이 본 유저에게 속하지 않으면 추천하라는 것
                        
#                             sim_list.append({'pik_id': key, 'similarity': 1})
#                             if len(sim_list) == 10:
#                                 break
#             return sim_list
        
    
#     else: 
#         result = get_most_similar_piks(pik_id, user_id, user_pik, piks_vec, piktitle_vec, num_link_by_pik, topk, threshold, piktitle_threshold, num_link_threshold)
#         return result
    

## This is modification of above to inlcude language condition
input_spec = Multipart(user_id=Text(), pik_id=Text())
@svc.api(input=input_spec, output=JSON())
def predict(user_id, pik_id) -> dict:
    
    similarity_dict = rec_pik_by_lang(pik_id, user_id, pik_status_dict, user_lang_dict_detected, user_lang_dict_userset, pik_lang_dict_detected, pik_lang_dict_userset, user_pik, piks_vec, piktitle_vec, num_link_by_pik,  topk=40, threshold=0.5, piktitle_threshold=0.5, num_link_threshold=1, min_user_link_num=5)
    return similarity_dict #sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)




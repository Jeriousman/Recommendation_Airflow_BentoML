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
runner = bentoml.transformers.get("user_recommender_model:latest").to_runner()

# runner.init_local()
# svc = bentoml.Service("feature-extraction-service", runners=[runner])
svc = bentoml.Service("user_recommender_bento", runners=[runner])


# with open("/opt/airflow/dags/data/pik_vec.json") as f:  ##avg_{data_type}_vec.pickle  avg_pik_vec.json or avg_user_vec.json
#     piks_vec = json.load(f)

    
# with open("/opt/airflow/dags/data/piktitle_emb_vec.json") as f:  ##pik_id_embeddings_vectors.json
#     piktitle_vec = json.load(f)

# with open("/opt/airflow/dags/data/num_link_by_pik.json") as f:
#     num_link_by_pik = json.load(f)

# df = pd.read_csv("/opt/airflow/dags/data/light_processed_data.csv")

# with open("/opt/airflow/dags/data/pik_lang_dict.json") as f:
#     pik_lang_dict = json.load(f)

with open("/opt/airflow/dags/data/user_vec.json") as f:  ##avg_{data_type}_vec.pickle  avg_pik_vec.json or avg_user_vec.json
    user_vec = json.load(f)
    
with open("/opt/airflow/dags/data/num_link_by_user.json") as f:
    num_link_by_user = json.load(f)
 
with open("/opt/airflow/dags/data/user_lang_dict.json") as f:
    user_lang_dict = json.load(f)


user_friends_list = pd.read_csv("/opt/airflow/dags/data/users_following.csv")


# sim_list = [103, 145, 11]
# user_id = '3408'
# topk=20
# threshold=0.7
# num_link_threshold =3
'''USER RECOMMENDATION LOGIC'''

def get_most_similar_users(user_id, user_vec, friends_list, num_link_by_user, topk, threshold, num_link_threshold):
    sim = list()
        
    for uid, vec in user_vec.items():
        
        thisSim = cosine_similarity(np.array(vec).reshape(1, -1), np.array(user_vec[user_id]).reshape(1, -1))
        sim.append((uid, thisSim[0][0]))
    
    full_ranked_similar_items = sorted(sim, key=lambda x: x[1], reverse=True) ##full similarity list
    ranked_similar_items = full_ranked_similar_items[:topk+1] ##only topk similarity list. 본유저도 들어가있기떄문에+1을해준다 
     
    
    sim_list = [] ##추천 candidate 추려내서 저장하는 리스트 
    for i in range(1, topk+1):  ###0번째는 본유저이기 때문에 빼놓고 한다. 
        if ranked_similar_items[i][1] > threshold:  
            if user_id != ranked_similar_items[i][0] and int(ranked_similar_items[i][0]) not in list(friends_list['followed_user_id'][friends_list['user_id'] == int(user_id)]) and int(ranked_similar_items[i][0]) not in list([sim_list[num]['user_id'] for num in range(len(sim_list))]): ##본 유저가 아니거나 본 유저가 팔로잉 있지 않은 유저면 추천하라는 것 
                if i >= 1:  
                    if ranked_similar_items[i-1][1] != ranked_similar_items[i][1]: ##유사도가 바로그다음으로높은것과비교했을때 현재유사도와같으면 같은내용의픽이나링크일테니 그건스킵하라는것
                        if num_link_by_user[ranked_similar_items[i][0]] >= num_link_threshold: ##픽안에 num_link_threshold 갯수이상 링크가 존재할때만 추천한다 
                            lottery = random()
                            
                            if lottery <= 0.88:
                                
                                sim_list.append({'user_id':ranked_similar_items[i][0], 'similarity':ranked_similar_items[i][1]})

                            # ## if we want piktitle_vec into consideration, we should use this code and modify other code accordingly.
                            # elif 0.7 < lottery <= 0.88:
                            #     pik_title_sim = cosine_similarity(np.array(piktitle_vec[ranked_similar_items[0][0]]).reshape(1, -1), np.array(piktitle_vec[ranked_similar_items[i][0]]).reshape(1, -1))[0][0]
                            #     if pik_title_sim >= piktitle_threshold: ##픽타이틀의 유사도도 threshold를 넘으면 그대로바로 추천리스트에들어간다  
                            #         sim_list.append({'user_id':ranked_similar_items[i][0], 'similarity':ranked_similar_items[i][1]})    

                            
                            elif 0.88 < lottery <= 1.0:
      
                                random_topk_rec_index = randint(0, topk)
                                if user_id != ranked_similar_items[random_topk_rec_index][0] and int(ranked_similar_items[random_topk_rec_index][0]) not in list(friends_list['followed_user_id'][friends_list['user_id'] == int(user_id)]) and ranked_similar_items[random_topk_rec_index][0] not in list([sim_list[num]['user_id'] for num in range(len(sim_list))]):
                                    sim_list.append({'user_id':ranked_similar_items[random_topk_rec_index][0], 'similarity':ranked_similar_items[random_topk_rec_index][1]})
                                
                            if len(sim_list) == 10:
                                    break        

    
    ## 만약 sim_list가비어있거나 None이면, 그보다 더유사도가 적지만 그래도괜찮을수있는것을 추천해준다. 
    if sim_list == None:
        print('There is no recommended piks for your pik for now')
    
    elif bool(sim_list):
        return sim_list
         
    
    # elif not bool(sim_list):  
    elif not bool(sim_list): ##sim_list에 아무것도 존재하지 않으면, second_threshold를 사용해서 더낮지만 그래도 차선인 추천을 해준다. 
        print('There is no recommended piks for your pik for now')
                 
        
                                    
    else:
        print('Hey, there are really not suitable recommendation for your pik for now. But we are working on it!')
                    






def get_most_similar_users_ko(user_id, user_vec, friends_list, num_link_by_user, topk, threshold, num_link_threshold):
    sim = list()
    
    for uid, vec in user_vec.items():
        if user_lang_dict[uid] == 'ko':
        # if pd.unique(data['language_code'][data['user_id'] == user_id])[0] == 'ko':
            thisSim = cosine_similarity(np.array(vec).reshape(1, -1), np.array(user_vec[user_id]).reshape(1, -1))
            sim.append((uid, thisSim[0][0]))

    full_ranked_similar_items = sorted(sim, key=lambda x: x[1], reverse=True) ##full similarity list
    ranked_similar_items = full_ranked_similar_items[:topk+1] ##only topk similarity list. 본픽도 들어가있기떄문에+1을해준다
    
    sim_list = [] ##추천 candidate 추려내서 저장하는 리스트 
    for i in range(1, topk+1):
        if ranked_similar_items[i][1] > threshold:
            if user_id != ranked_similar_items[i][0] and int(ranked_similar_items[i][0]) not in list(friends_list['followed_user_id'][friends_list['user_id'] == int(user_id)]) and int(ranked_similar_items[i][0]) not in list([sim_list[num]['user_id'] for num in range(len(sim_list))]): ##본 유저가 아니거나 본 유저가 팔로잉 있지 않은 유저면 추천하라는 것
                if i >= 1:  
                    if ranked_similar_items[i-1][1] != ranked_similar_items[i][1]: ##유사도가 바로그다음으로높은것과비교했을때 현재유사도와같으면 같은내용의픽이나링크일테니 그건스킵하라는것
                        if num_link_by_user[ranked_similar_items[i][0]] >= num_link_threshold: ##픽안에 num_link_threshold 갯수이상 링크가 존재할때만 추천한다 
                            lottery = random()
                            if lottery <= 0.88:
                                
                                sim_list.append({'user_id':ranked_similar_items[i][0], 'similarity':ranked_similar_items[i][1]})

                            ### if we want piktitle_vec into consideration, we should use this code and modify other code accordingly.
                            # elif 0.7 < lottery <= 0.88:
                            #     pik_title_sim = cosine_similarity(np.array(piktitle_vec[ranked_similar_items[0][0]]).reshape(1, -1), np.array(piktitle_vec[ranked_similar_items[i][0]]).reshape(1, -1))[0][0]
                            #     if pik_title_sim >= piktitle_threshold: ##픽타이틀의 유사도도 threshold를 넘으면 그대로바로 추천리스트에들어간다  
                            #         sim_list.append({'user_id':ranked_similar_items[i][0], 'similarity':ranked_similar_items[i][1]})    

                            
                            elif 0.88 < lottery <= 1.0:
      
                                random_topk_rec_index = randint(0, topk)
                                if user_id != ranked_similar_items[random_topk_rec_index][0] and int(ranked_similar_items[random_topk_rec_index][0]) not in list(friends_list['followed_user_id'][friends_list['user_id'] == int(user_id)]) and ranked_similar_items[random_topk_rec_index][0] not in list([sim_list[num]['user_id'] for num in range(len(sim_list))]):
                                    sim_list.append({'user_id':ranked_similar_items[random_topk_rec_index][0], 'similarity':ranked_similar_items[random_topk_rec_index][1]})
                                
                            if len(sim_list) == 10:
                                    break  
                            
        
    
    ## 만약 sim_list가비어있거나 None이면, 그보다 더유사도가 적지만 그래도괜찮을수있는것을 추천해준다. 
    if sim_list == None:
        print('안타깝게도 현재 최적의 유저 추천이 어려운 상황이네요 ㅠㅠ')
    
    elif bool(sim_list):
        return  sim_list   

    # elif not bool(sim_list):  
    elif not bool(sim_list): ##sim_list에 아무것도 존재하지 않으면, second_threshold를 사용해서 더낮지만 그래도 차선인 추천을 해준다. 
                 
        print('안타깝게도 현재 최적의 유저 추천이 어려운 상황이네요 ㅠㅠ')
                                    
    else:
        print('정말 죄송하지만 현재 최적의 유저 추천이 어려운 상황입니다 ㅠㅠ! 열일하고 있으니 조금만 기다려 주세요!')
              
                

 
def get_most_similar_users_en(user_id, user_vec, friends_list, num_link_by_user, topk, threshold, num_link_threshold):
    sim = list()
        
    for uid, vec in user_vec.items():
        if user_lang_dict[uid] == 'en':
        # if pd.unique(data['language_code'][data['user_id'] == user_id])[0] == 'en':
        
            thisSim = cosine_similarity(np.array(vec).reshape(1, -1), np.array(user_vec[user_id]).reshape(1, -1))
            sim.append((uid, thisSim[0][0]))

    full_ranked_similar_items = sorted(sim, key=lambda x: x[1], reverse=True) ##full similarity list
    ranked_similar_items = full_ranked_similar_items[:topk+1] ##only topk similarity list. 본픽도 들어가있기떄문에+1을해준다
     
    
    sim_list = [] ##추천 candidate 추려내저서 저장하는 리스트 
    for i in range(1, topk+1):
        if ranked_similar_items[i][1] > threshold:  
            if user_id != ranked_similar_items[i][0] and int(ranked_similar_items[i][0]) not in list(friends_list['followed_user_id'][friends_list['user_id'] == int(user_id)]) and int(ranked_similar_items[i][0]) not in list([sim_list[num]['user_id'] for num in range(len(sim_list))]): ##본 유저가 아니거나 본 유저가 팔로잉 있지 않은 유저면 추천하라는 것: ##본픽이 아니라면 추천하라는 뜻
                if i >= 1:  
                    if ranked_similar_items[i-1][1] != ranked_similar_items[i][1]: ##유사도가 바로그다음으로높은것과비교했을때 현재유사도와같으면 같은내용의픽이나링크일테니 그건스킵하라는것
                        if num_link_by_user[ranked_similar_items[i][0]] >= num_link_threshold: ##픽안에 num_link_threshold 갯수이상 링크가 존재할때만 추천한다 
                            lottery = random()
                            if lottery <= 0.88:
                                
                                sim_list.append({'user_id':ranked_similar_items[i][0], 'similarity':ranked_similar_items[i][1]})

                            ### if we want piktitle_vec into consideration, we should use this code and modify other code accordingly.
                            # elif 0.7 < lottery <= 0.88:
                            #     pik_title_sim = cosine_similarity(np.array(piktitle_vec[ranked_similar_items[0][0]]).reshape(1, -1), np.array(piktitle_vec[ranked_similar_items[i][0]]).reshape(1, -1))[0][0]
                            #     if pik_title_sim >= piktitle_threshold: ##픽타이틀의 유사도도 threshold를 넘으면 그대로바로 추천리스트에들어간다  
                            #         sim_list.append({'user_id':ranked_similar_items[i][0], 'similarity':ranked_similar_items[i][1]})    

                            
                            elif 0.88 < lottery <= 1.0:
      
                                random_topk_rec_index = randint(0, topk)
                                if user_id != ranked_similar_items[random_topk_rec_index][0] and int(ranked_similar_items[random_topk_rec_index][0]) not in list(friends_list['followed_user_id'][friends_list['user_id'] == int(user_id)]) and ranked_similar_items[random_topk_rec_index][0] not in list([sim_list[num]['user_id'] for num in range(len(sim_list))]):
                                    sim_list.append({'user_id':ranked_similar_items[random_topk_rec_index][0], 'similarity':ranked_similar_items[random_topk_rec_index][1]})
                                
                            if len(sim_list) == 18:
                                    break  
        
    
    ## 만약 sim_list가비어있거나 None이면, 그보다 더유사도가 적지만 그래도괜찮을수있는것을 추천해준다. 
    if sim_list == None:
        print('There is no recommended users for you for now')
    
    elif bool(sim_list):
        return  sim_list 


    # elif not bool(sim_list):  
    elif not bool(sim_list): ##sim_list에 아무것도 존재하지 않으면, second_threshold를 사용해서 더낮지만 그래도 차선인 추천을 해준다. 
        print('There is no recommended users for you for now')

    else:
        print('Hey, there are really not suitable users to recommend for you for now. But we are working on it!')



def rec_user_by_lang(user_id, user_vec, friends_list, num_link_by_user, topk, threshold, num_link_threshold):    

    if user_lang_dict[user_id] == 'ko':
        result = get_most_similar_users_ko(user_id, user_vec, friends_list, num_link_by_user, topk, threshold, num_link_threshold)
        return result

    elif user_lang_dict[user_id] == 'en':
        result = get_most_similar_users_en(user_id, user_vec, friends_list, num_link_by_user, topk, threshold, num_link_threshold)
        return result

    else: 
        result = get_most_similar_users(user_id, user_vec, friends_list, num_link_by_user, topk, threshold, num_link_threshold)
        return result
        # return inspect.signature(rec_user_by_lang)
    
# zz = rec_user_by_lang(96, user_vec, user_friends, num_link_by_user, link_cat_pik, 10, 0.95, 0.89, 3)



## This is modification of above to inlcude language condition
input_spec = Multipart(user_id=Text())
@svc.api(input=input_spec, output=JSON())
def predict(user_id) -> dict:

    similarity_dict = rec_user_by_lang(user_id, user_vec, user_friends_list, num_link_by_user, topk=30, threshold=0.7, num_link_threshold=4)
    return similarity_dict



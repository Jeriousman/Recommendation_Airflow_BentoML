#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:07:42 2022

@author: hojun
"""
from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pandas as pd
import pickle
import json
import numpy as np
import lshashpy3

def load_tokenizer_and_model(tokenizer_name, model_name):
    
    '''
    embedding을사용하려면 nli모델을사용해야한다
    '''
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
    return tokenizer, model
    


##링크 타이틀과 픽 타이틀 엠베딩이 만들어졌지만 픽을 구성하는 엠베딩을 아직 계산하지는 않았기 때문에 링크를 픽별로 그룹화 해줌 
##유저/픽별로 모든 링크를수집해서 딕셔너리화하기        
def get_links_by(dataframe, groupby, groupwhat):
    
    groupby_link = dataframe[[groupby, groupwhat]].groupby(groupby)[groupwhat].apply(lambda x: x.tolist()).to_dict() #한 유저가 몇개의 voca(빌딩과카드번호로를 틀렸는지 다 보여준다 #groupby는 유저로그룹묶어주고 voca에대한값을보여달라는것
    return groupby_link  


#################### 두개이상틀린것에대해서 유사도측정####################################

def get_vectors(first_map, second_map):
    first_vec  = dict()
    for uid, links in first_map.items(): #voca와 voca안에 있는 text로 나누어줌 
        temp = list()
        for element in links:
            try:
                temp.append(second_map[element]) #voca에 있는 text의 각 단어들을 모두 append 하는 것 
            except KeyError:
                pass
        first_vec[uid] = np.mean(temp, axis=0)  #append된 모든 단어들의 값을 mean으로 평균내어 주는 것 
    
    return first_vec


def train_save_lsh(hash_size, input_dim, num_hashtables, matrices_filename, hashtable_filename, link_vector):
    '''LSH link recommender logic. 왜냐하면 위의 일반 link recommender는 너무 느리기 때문에 LSH를 사용해야 한다'''
    lsh = lshashpy3.LSHash(hash_size=hash_size, input_dim=input_dim, num_hashtables=num_hashtables,
            storage_config={ 'dict': None },
            matrices_filename= matrices_filename,  ##'weights.npz'
            hashtable_filename= hashtable_filename, ##'hash.npz' 
            overwrite=False) 


    ##index한다는 것은 link_vec들을 모두 LSH에 등록한다는 것. 이제 이 등록된 것들을 query 벡터랑 비교하기만 하면된다.
    for link_id, link_vector in tqdm(zip(link_vector.keys(), link_vector.values())):
        lsh.index(link_vector, extra_data = link_id)      

    ##나중에도 쓰고 싶으면 세이빙을 하는 것이 좋다.
    lsh.save()



# def calculate_emb(processed_data_path, tokenizer_name, model_name, dataloader, embeddings='link_emb'):
def calculate_emb(**kwargs):
    default_path = kwargs.get('default_path', '/opt/airflow/dags')
    processed_data_path = kwargs.get('processed_data_path', '/opt/airflow/dags/data/processed_data.csv')   
    tokenizer_name = kwargs.get('tokenizer_name', 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking')   
    model_name = kwargs.get('model_name', 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking')   
    dataloader_path = kwargs.get('dataloader_path', '/opt/airflow/dags/data/link_title_dataloader.pickle')   
    which_emb = kwargs.get('which_emb', 'linktitle_emb')  
    link_rec_on = kwargs.get('link_rec_on', False)  
    device = kwargs.get('device', 'cpu')   

    processed_data = pd.read_csv(processed_data_path)
    tokenizer, model = load_tokenizer_and_model(tokenizer_name, model_name)
    with open(dataloader_path, 'rb') as f:
        dataloader = pickle.load(f)
    
    model.eval()
    mean_pooled_total = []
    if model_name == 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking':
            
        
        for batch in tqdm(dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_att_masks = batch
                
            # Run the text through BERT, and collect all of the hidden states produced
            # from all 12 layers. 
         
            with torch.no_grad():
                    outputs = model(b_input_ids, attention_mask = b_att_masks) ##For distilbert we dont need token_type_ids
                    # outputs.keys()
                    embeddings = outputs.last_hidden_state
                    mask = b_att_masks.unsqueeze(-1).expand(embeddings.size()).float()
                    masked_embeddings = embeddings * mask
                    summed_emb = torch.sum(masked_embeddings, 1)
                    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
                    mean_pooled_emb = summed_emb / summed_mask
                    # mean_pooled_emb = F.normalize(mean_pooled_emb, p=2, dim=1)

            mean_pooled_total.append(mean_pooled_emb)
            
    else:
        for batch in tqdm(dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_att_masks = batch
                
            # Run the text through BERT, and collect all of the hidden states produced
            # from all 12 layers. 
         
            with torch.no_grad():
                    outputs = model(b_input_ids, attention_mask = b_att_masks, token_type_ids=None)
                    outputs.keys()
                    # pooled_outputs = outputs[1]  ##(last_hidden_state, pooler_output, hidden_states[optional], attentions[optional])
                    embeddings = outputs.last_hidden_state
                    mask = b_att_masks.unsqueeze(-1).expand(embeddings.size()).float()
                    masked_embeddings = embeddings * mask
                    summed_emb = torch.sum(masked_embeddings, 1)
                    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
                    mean_pooled_emb = summed_emb / summed_mask
            
            mean_pooled_total.append(mean_pooled_emb)
            
    if which_emb == 'linktitle_emb':
        link_final_pred = torch.cat(mean_pooled_total, 0).detach().cpu().numpy()
        link_vectors = dict(zip(processed_data.link_id, link_final_pred))  ##link title vectors

        if link_rec_on:
            '''link_vec을 저장하기 위해서는 아래 코드들을 언코멘트 해준다'''
            # 아래 코드는 BentoML에서 실행하기 위해 중요하다. Bento는 json데이터에 가장 친숙하기 때문에 왠만해선 json을 쓰도록하자
            link_vectors_tolist = {str(k): v.tolist() for k, v in link_vectors.items()}
            with open(f"{default_path}/data/{which_emb}_vec.json", "w") as f: ##2G가까이되는 큰 데이터이기 때문에 왠만하면 세이브하지말자
                json.dump(link_vectors_tolist, f)
    
            train_save_lsh(hash_size=20,
                            input_dim=768,
                            num_hashtables=10,
                            matrices_filename=f'{default_path}/data/lsh_matrices_filename.npz',
                            hashtable_filename=f'{default_path}/data/lsh_hashtables_filename.npz',
                            link_vector=link_vectors)      

        ## keys: pik, values: link_id ##pik_id로 link를 그룹화해라라는뜻
        ##pik추천을 위한 것
        pik_link = get_links_by(processed_data, 'pik_id', 'link_id')
        pik_vec = get_vectors(pik_link, link_vectors) #유저가 틀린 문장들이계산되었는데 그문장들을 모두모아서에버리지하는것. 유저가틀린모든문장을모아서평균계산
        ## 아래 코드는 BentoML에서 실행하기 위해 중요하다. Bento는 json데이터에 가장 친숙하기 때문에 왠만해선 json을 쓰도록하자
        pik_vec_tolist = {str(k):v.tolist() for k,v in pik_vec.items()}
        with open(f"{default_path}/data/pik_vec.json", "w") as f:
            json.dump(pik_vec_tolist, f)
        
        ## 아래 코드는 BentoML에서 실행하기 위해 중요하다. Bento는 json데이터에 가장 친숙하기 때문에 왠만해선 json을 쓰도록하자
        ##pik을 기준으로 link를 딕셔너리로 정렬시키고 그것을 json으로 저장하라 
        num_link_by_pik = processed_data.groupby('pik_id')['link_id'].nunique().sort_values(ascending=False)
        num_link_by_pik = {str(key):value for key,value in num_link_by_pik.items()} ##딕셔너리화한다
        with open(f'{default_path}/data/num_link_by_pik.json', 'w') as f:
            json.dump(num_link_by_pik, f)


        user_pik = get_links_by(processed_data, 'user_id', 'pik_id')
        with open(f'{default_path}/data/user_pik.json', 'w') as f:
            json.dump(user_pik, f)

        user_link = get_links_by(processed_data, 'user_id', 'link_id')
        with open(f'{default_path}/data/user_link.json', 'w') as f:
            json.dump(user_link, f)
        

        ##유저추천을 위한 것
        num_link_by_user = processed_data.groupby('user_id')['link_id'].nunique().sort_values(ascending=False)
        num_link_by_user = {str(key):value for key,value in num_link_by_user.items()} ##딕셔너리화한다 
        with open(f'{default_path}/data/num_link_by_user.json', 'w') as f:
            json.dump(num_link_by_user, f)

        user_link = get_links_by(processed_data, 'user_id', 'link_id')
        ##for user-rec
        user_vec = get_vectors(user_link, link_vectors)
        user_vec_tolist = {str(k):v.tolist() for k,v in user_vec.items()}
        with open(f"{default_path}/data/user_vec.json", "w") as f:
            json.dump(user_vec_tolist, f)
        
        
    elif which_emb == 'piktitle_emb':
        pik_final_pred = torch.cat(mean_pooled_total, 0).detach().cpu().numpy()
        piktitle_vectors = dict(zip(processed_data.pik_id, pik_final_pred))  ##pik title vectors
    
        ## 아래 코드는 BentoML에서 실행하기 위해 중요하다. Bento는 json데이터에 가장 친숙하기 때문에 왠만해선 json을 쓰도록하자
        piktitle_vectors_tolist = {str(k): v.tolist() for k, v in piktitle_vectors.items()}
        with open(f"{default_path}/data/{which_emb}_vec.json", "w") as f: ##2G가까이되는 큰 데이터이기 때문에 왠만하면 세이브하지말자
            json.dump(piktitle_vectors_tolist, f)
    
    del tokenizer
    del model     ##freeing space
    torch.cuda.empty_cache() ##GPU clearinig up

   
    
    
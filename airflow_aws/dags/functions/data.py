#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:31:37 2022

@author: hojun
"""

import os
import pandas as pd
from typing import List
import pickle
import re
import pandas as pd
from typing import List
import json
def filtering_users(data, user_colname, user_list: List):
    '''
    Parameters
    ----------
    data : Pandas dataframe
        DESCRIPTION.
    user_colname : string
        DESCRIPTION.
    user_list : List
        DESCRIPTION.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    '''
    for user in user_list:
        data = data[data[user_colname] != user]
    return data



def preprocess(document: str) -> str:
    # remove URL pattern 
    # 안녕https://m.naver.com하세요 -> 안녕하세요
    # pattern = r'(http|https|ftp)://(?:[-\w.]|[\w/]|[\w\?]|[\w:])+'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove tag pattern
    # # 안녕<tag>하세요 -> 안녕하세요
    # pattern = r'<[^>]*>'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove () and inside of ()
    # # 안녕(parenthese)하세요 -> 안녕하세요
    # pattern = r'\([^)]*\)'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove [] and inside of []
    # # 안녕[parenthese]하세요 -> 안녕하세요
    # pattern = r'\[[^\]]*\]'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove special chars without comma and dot
    # # 안녕!!@@하세요, 저는 !@#호준 입니다. -> 안녕하세요, 저는 호준 입니다.
    # pattern = r'[^\w\s - .|,]'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove list characters
    # # 안녕12.1하세요 -> 안녕하세요
    # pattern = r'[0-9]*[0-9]\.[0-9]*'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove korean consonant and vowel
    # # 안녕ㅏ하ㅡㄱ세요 -> 안녕하세요
    # pattern = r'([ㄱ-ㅎㅏ-ㅣ]+)'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove chinese letter
    # # 안녕山하세요 -> 안녕하세요
    # pattern = r'[一-龥]*'
    # document = re.sub(pattern=pattern, repl='', string=document)
    
    ##https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=realuv&logNo=220699272999
    pattern =r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9 ]' 
    # pattern =r'[^가-힣ㄱ-ㅎㅏ-ㅣ]' ##숫자와 영어를뺴고싶은경우
    document = re.sub(pattern=pattern, repl=' ', string=document)
    
    # 영어 소문자로 변환
    # document = document.lower()

    # remove empty space
    document = document.strip()

    # make empty spcae size as only one
    document = ' '.join(document.split())
    
    return document

##invalid 데이터 필터링아웃하기
def filter_data(dataframe, cols):
    '''
    Purpose:
        filtering non-character values
        filtering invalid characters
    '''
    dataframe.dropna(axis=0, how='any', inplace=True, subset = cols)
    # dataframe = dataframe.dropna(subset= cols)
    for col in cols:
        dataframe[col] = [preprocess(str(text)) for text in dataframe[col]]
        dataframe = dataframe[dataframe[col] != ' ']
        dataframe = dataframe[dataframe[col] != 'nan']        
        dataframe = dataframe[dataframe[col] != ''] 
        dataframe = dataframe[dataframe[col] != '\n'] 
    return dataframe

def check_data(**kwargs):
    # data = ti.xcom_pull(key='data', task_ids=['raw_data_preprocess'])
    df = kwargs.get('df', '/home/ubuntu/airflow/dags/data/link_cat_pik.csv') 
    processed_data_path = kwargs.get('processed_data_path', '/home/ubuntu/airflow/dags/data/processed_data.csv') 
    data = pd.read_csv(df)
    # data=data.dropna(subset=['link_title', 'pik_title'])
    data=data.dropna(subset=['link_title'])
    data=data.dropna(subset=['pik_title'])
    
    data.to_csv(processed_data_path, index=False)
    print('processed_data shape is: ', data.shape)
    


def raw_data_preprocess(**kwargs):
    
    
    '''
    raw_data_path:
    artifical_user_list_path: Non-natural users should be removed from recommendation.
    processed_data_saving_path: A path to save processed pandas dataframe that will be constantly used.
    '''
    
    
    
    #model_name = kwargs.get('model_name', 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking')
    path = kwargs.get('path', '/home/ubuntu/airflow/dags/data')    
    
    linkhub = pd.read_csv(f'{path}/linkhub_link.csv')
    piks = pd.read_csv(f'{path}/piks_pik.csv')
    catego = pd.read_csv(f'{path}/piks_category.csv')
    user_language = pd.read_csv(f'{path}/users_user.csv')
    user_friends = pd.read_csv(f'{path}/users_following.csv')
    

    ##processing with friends list
    user_friends.rename(columns = {'from_user_id': 'user_id', 'to_user_id':'followed_user_id'}, inplace=True)
    user_friends = user_friends[user_friends['is_deleted'] == False]
    user_friends.to_csv(f'{path}/users_following.csv', index=False)


    with open(f'{path}/artificial_user_list', 'rb') as f:
        artificial_users = pickle.load(f)
        
    
    
    linkhub.rename(columns = {'title':'link_title'}, inplace=True)
    piks.rename(columns = {'title':'pik_title'}, inplace=True)
    catego.rename(columns = {'title': 'cat_title'}, inplace = True)


    ##category 테이블에 user id가 있기 때문에 그 아이디와 유저의 언어설정환경을 조인한다.
    catego = pd.merge(catego, user_language, how = 'inner', left_on ='user_id', right_on='id', suffixes=('', '_user'))


    ##pik_info와 cat_info 병합
    piks_cats = pd.merge(catego, piks, how = 'inner', left_on = 'pik_id', right_on = 'id', suffixes=('_cat', '_pik'))
    piks_cats.columns
    filtered_pik_cat  = piks_cats[(piks_cats['status'] == 'public') & (piks_cats['is_draft'] == False)]




    ##이렇게해서 public이고 draft가 아닌 픽만남게된다
    link_cat_pik = pd.merge(linkhub, filtered_pik_cat, how='inner', left_on='category_id', right_on='id_cat', suffixes=('_link', '_catpik'))
    link_cat_pik.columns
    link_cat_pik.dropna(subset=['link_title', 'pik_title'], how='any', inplace=True)

    link_cat_pik.rename(columns={'id':'link_id', 'created':'link_create_time'}, inplace=True)
    link_cat_pik.sort_values(['user_id', 'category_id', 'link_id'], ascending = True, inplace=True)
    

    # from glob import glob
    slug1 = link_cat_pik['slug'][link_cat_pik['pik_id'] == 3085].iloc[0]
    slug2 = link_cat_pik['slug'][link_cat_pik['pik_id'] == 3186].iloc[0]
    slug3 = link_cat_pik['slug'][link_cat_pik['pik_id'] == 3188].iloc[0]
    
    ##특정 string이 포함된 행을 df에서찾기
    dummy1 = link_cat_pik[link_cat_pik['slug'].str.contains(slug1)]
    dummy2 = link_cat_pik[link_cat_pik['slug'].str.contains(slug2)]
    dummy3 = link_cat_pik[link_cat_pik['slug'].str.contains(slug3)]
    
    ## 추출한 행을 df에서제거하기
    link_cat_pik = link_cat_pik[~link_cat_pik.index.isin(dummy1.index)]
    link_cat_pik = link_cat_pik[~link_cat_pik.index.isin(dummy2.index)]
    link_cat_pik = link_cat_pik[~link_cat_pik.index.isin(dummy3.index)]
    
    ##제거된것을확인
    link_cat_pik[link_cat_pik['slug'] == slug1]
    link_cat_pik[link_cat_pik['slug'] == slug2]
    link_cat_pik[link_cat_pik['slug'] == slug3]
    
    link_cat_pik.drop(['description', 'memo', 'url', 'is_draft_link', 'link_create_time', 'id_cat', 'created_cat', 'id_user', 'id_pik', 'slug', 'language', 'is_draft_catpik', 'created_pik'], axis=1, inplace=True)
    link_cat_pik = link_cat_pik.astype({'pik_id' :'int', 'link_id':'int', 'category_id':'int'})    
    
    link_cat_pik = filtering_users(link_cat_pik, 'user_id', artificial_users)  
    
    link_cat_pik = filter_data(link_cat_pik,  ['link_title'])
    link_cat_pik = filter_data(link_cat_pik,  ['pik_title'])
    
    # link_cat_pik.dropna(axis=0, how='any', inplace=True, subset = 'link_title')
    # link_cat_pik.dropna(axis=0, how='any', inplace=True, subset = 'pik_title')
    # ti.xcom_push(key='data', value =  link_cat_pik)
    
        
    
    user_lang_dict = {k:v for k,v in zip(link_cat_pik['user_id'], link_cat_pik['language_code'])}
    pik_lang_dict = {k:v for k,v in zip(link_cat_pik['pik_id'], link_cat_pik['language_code'])}
    link_lang_dict = {k:v for k,v in zip(link_cat_pik['link_id'], link_cat_pik['language_code'])}
    
    with open(f"{path}/user_lang_dict.json", "w") as f:  ##For bento_service.py
        json.dump(user_lang_dict, f)
        
    with open(f"{path}/pik_lang_dict.json", "w") as f:   ##For bento_service.py
        json.dump(pik_lang_dict, f)

    with open(f"{path}/link_lang_dict.json", "w") as f:   ##For bento_service.py
        json.dump(link_lang_dict, f)
    




    
    
    link_cat_pik.to_csv(f'{path}/link_cat_pik.csv', index=False)
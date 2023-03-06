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
import psycopg2


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
    df = kwargs.get('df', '/opt/airflow/dags/data/link_cat_pik.csv') 
    processed_data_path = kwargs.get('processed_data_path', '/opt/airflow/dags/data/processed_data.csv') 
    data = pd.read_csv(df)
    # data=data.dropna(subset=['link_title', 'pik_title'])
    data=data.dropna(subset=['link_title'])
    data=data.dropna(subset=['pik_title'])
    
    data.to_csv(processed_data_path, index=False)
    print('processed_data shape is: ', data.shape)
    


def db_data_fetching(**kwargs):
    
    ##운영 DB
    default_path = kwargs.get('default_path', '/opt/airflow/dags/data')
    hostname = kwargs.get('hostname', 'prod-back.c5dkkbujxodg.ap-northeast-2.rds.amazonaws.com')    
    dbname = kwargs.get('dbname', 'pikurate') 
    username = kwargs.get('username', 'postgres')
    password = kwargs.get('password', 'postgres')
    portnumber = kwargs.get('portnumber', 5432)
    
    
    ##개발 DB
    # default_path = kwargs.get('default_path', '/opt/airflow/dags/data')
    # hostname = kwargs.get('hostname', 'dev-postgres.c5dkkbujxodg.ap-northeast-2.rds.amazonaws.com')    
    # dbname = kwargs.get('dbname', 'pikurateqa') 
    # username = kwargs.get('username', 'postgres')
    # password = kwargs.get('password', 'wXVcn64CZsdM27')
    # portnumber = kwargs.get('portnumber', 5432)







    conn = psycopg2.connect(host=hostname, dbname=dbname, user=username, password=password, port=portnumber)
    cur = conn.cursor()

    # export to csv
    fid = open(f'{default_path}/piks_category.csv', 'w')
    sql = "COPY (SELECT id, title, pik_id, user_id, created, is_deleted FROM piks_category) TO STDOUT WITH CSV HEADER"
    cur.copy_expert(sql, fid)
    fid.close()


    # export to csv
    fid = open(f'{default_path}/piks_pik.csv', 'w')
    sql = "COPY (SELECT id, slug, title, status, language, is_draft, created, is_deleted FROM piks_pik) TO STDOUT WITH CSV HEADER"
    cur.copy_expert(sql, fid)
    fid.close()


    # export to csv
    fid = open(f'{default_path}/users_user.csv', 'w')
    sql = "COPY (SELECT id, language_code FROM users_user) TO STDOUT WITH CSV HEADER"
    cur.copy_expert(sql, fid)
    fid.close()


    # export to csv
    fid = open(f'{default_path}/linkhub_link.csv', 'w')
    sql = "COPY (SELECT id, category_id, title, description, memo, url, is_draft, created, is_deleted FROM linkhub_link) TO STDOUT WITH CSV HEADER"
    cur.copy_expert(sql, fid)
    fid.close()


    # export to csv
    fid = open(f'{default_path}/users_following.csv', 'w')
    sql = "COPY (SELECT id, from_user_id, to_user_id, is_deleted FROM users_following) TO STDOUT WITH CSV HEADER"
    cur.copy_expert(sql, fid)
    fid.close()






def raw_data_preprocess(**kwargs):
    
    
    '''
    raw_data_path:
    artifical_user_list_path: Non-natural users should be removed from recommendation.
    processed_data_saving_path: A path to save processed pandas dataframe that will be constantly used.
    '''
    
    
    
    #model_name = kwargs.get('model_name', 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking')
    # path = '/home/hojun/python/rec_airflow_aws/dags/data'   
    path = kwargs.get('path', '/opt/airflow/dags/data')  

    
    linkhub = pd.read_csv(f'{path}/linkhub_link.csv')
    piks = pd.read_csv(f'{path}/piks_pik.csv')
    catego = pd.read_csv(f'{path}/piks_category.csv')
    user_language = pd.read_csv(f'{path}/users_user.csv')
    user_friends = pd.read_csv(f'{path}/users_following.csv')
    
    # linkhub = pd.read_csv('/home/hojun/temp/data_development/linkhub_link.csv')
    # piks = pd.read_csv('/home/hojun/temp/data_development/piks_pik.csv')
    # catego = pd.read_csv('/home/hojun/temp/data_development/piks_category.csv')
    # user_language = pd.read_csv('/home/hojun/temp/data_development/users_user.csv')
    # user_friends = pd.read_csv('/home/hojun/temp/data_development/users_following.csv')
    # with open('/home/hojun/temp/data_development/artificial_user_list', 'rb') as f:
    #     artificial_users = pickle.load(f)
        

    ##processing with friends list
    user_friends.rename(columns = {'from_user_id': 'user_id', 'to_user_id':'followed_user_id'}, inplace=True)
    user_friends = user_friends[(user_friends['is_deleted'] == 'f') | (user_friends['is_deleted'] == False)]  #'f' instead of False
    user_friends.to_csv(f'{path}/users_following.csv', index=False)


    with open(f'{path}/artificial_user_list', 'rb') as f:
        artificial_users = pickle.load(f)
        
    
    
    linkhub.rename(columns = {'title':'link_title'}, inplace=True)
    linkhub = linkhub[(linkhub['is_deleted'] == 'f') | (linkhub['is_deleted'] == False)]

    
    piks.rename(columns = {'title':'pik_title'}, inplace=True)
    piks = piks[(piks['is_deleted'] == 'f') | (piks['is_deleted'] == False)]
    
    
    catego.rename(columns = {'title': 'cat_title'}, inplace = True)
    catego = catego[(catego['is_deleted'] == 'f') | (catego['is_deleted'] == False)]
    

    ##category 테이블에 user id가 있기 때문에 그 아이디와 유저의 언어설정환경을 조인한다.
    catego = pd.merge(catego, user_language, how = 'inner', left_on ='user_id', right_on='id', suffixes=('', '_user'))


    ##pik_info와 cat_info 병합
    piks_cats = pd.merge(catego, piks, how = 'inner', left_on = 'pik_id', right_on = 'id', suffixes=('_cat', '_pik'))
    piks_cats.columns
    # filtered_pik_cat  = piks_cats[(piks_cats['status'] == 'public') & (piks_cats['is_draft'] == 'f')] #'f' instead of False
    public_private_filtered_pik_cat = piks_cats[piks_cats['is_draft'] == 'f']




    ##이렇게해서 public이고 draft가 아닌 픽만남게된다
    # link_cat_pik = pd.merge(linkhub, filtered_pik_cat, how='inner', left_on='category_id', right_on='id_cat', suffixes=('_link', '_catpik'))
    # link_cat_pik.columns
    # link_cat_pik.dropna(subset=['link_title', 'pik_title'], how='any', inplace=True)
    # link_cat_pik.rename(columns={'id':'link_id', 'created':'link_create_time'}, inplace=True)
    # link_cat_pik.sort_values(['user_id', 'category_id', 'link_id'], ascending = True, inplace=True)
    
    
    ##공개 비공개 픽 모두지만 na값은 없앴다 (즉 링크가 없는 빈 픽이나 유저는 없앴다)
    link_cat_pik = pd.merge(linkhub, public_private_filtered_pik_cat, how='inner', left_on='category_id', right_on='id_cat', suffixes=('_link', '_catpik'))
    link_cat_pik.rename(columns={'id':'link_id', 'created':'link_create_time'}, inplace=True)
    link_cat_pik.sort_values(['user_id', 'category_id', 'link_id'], ascending = True, inplace=True)  
    

    '''
    일반적으로 하던, 모든 na값 빼고 private pik빼고 inner 조인도 한 그런 값 
    '''
    
    def dropNa_dropArtificialUser(data, artificial_users, user_language_df, dropna=True, drop_artificial=True, integering=True):
        if drop_artificial:
        # from glob import glob
            slug1 = data['slug'][data['pik_id'] == 3085].iloc[0]
            slug2 = data['slug'][data['pik_id'] == 3186].iloc[0]
            slug3 = data['slug'][data['pik_id'] == 3188].iloc[0]
            
            ##특정 string이 포함된 행을 df에서찾기
            dummy1 = data[data['slug'].str.contains(slug1)]
            dummy2 = data[data['slug'].str.contains(slug2)]
            dummy3 = data[data['slug'].str.contains(slug3)]
            
            ## 추출한 행을 df에서제거하기
            data = data[~data.index.isin(dummy1.index)]
            data = data[~data.index.isin(dummy2.index)]
            data = data[~data.index.isin(dummy3.index)]
            
            ##제거된것을확인
            data[data['slug'] == slug1]
            data[data['slug'] == slug2]
            data[data['slug'] == slug3]
            
            data.drop(['description', 'memo', 'url', 'is_draft_link', 'link_create_time', 'id_cat', 'created_cat', 'id_user', 'id_pik', 'slug', 'language', 'is_draft_catpik', 'created_pik'], axis=1, inplace=True)
            
        if integering:
            data = data.astype({'pik_id' :'int', 'link_id':'int', 'category_id':'int'})    
        
        if dropna:
            data = filtering_users(data, 'user_id', artificial_users)      
            data = filter_data(data,  ['link_title'])
            data = filter_data(data,  ['pik_title'])
            
        # link_cat_pik.dropna(axis=0, how='any', inplace=True, subset = 'link_title')
        # link_cat_pik.dropna(axis=0, how='any', inplace=True, subset = 'pik_title')
        # ti.xcom_push(key='data', value =  link_cat_pik)
    
        
    
        user_lang_dict_userset = {str(k):str(v) for k,v in zip(user_language_df['id'], user_language_df['language_code'])} ##모든유저를 검색할 수 있도록
        pik_lang_dict_userset = {k:v for k,v in zip(data['pik_id'], data['language_code'])}
        link_lang_dict_userset = {k:v for k,v in zip(data['link_id'], data['language_code'])}
        pik_status_dict = {k:v for k,v in zip(data['pik_id'], data['status'])}
        
        linkid_title_dict = {k:v for k,v in zip(data['link_id'], data['link_title'])}
        pikid_title_dict = {k:v for k,v in zip(data['pik_id'], data['pik_title'])}
        
        return user_lang_dict_userset, pik_lang_dict_userset, link_lang_dict_userset, pik_status_dict, linkid_title_dict, pikid_title_dict, data
        # return pik_status_dict, linkid_title_dict, pikid_title_dict, data
    

    # pik_status_dict, linkid_title_dict, pikid_title_dict, link_cat_pik = dropNa_dropArtificialUser(link_cat_pik, artificial_users, user_language, dropna=True, drop_artificial=True, integering=True)
    user_lang_dict_userset, pik_lang_dict_userset, link_lang_dict_userset, pik_status_dict, linkid_title_dict, pikid_title_dict, link_cat_pik = dropNa_dropArtificialUser(link_cat_pik, artificial_users, user_language, dropna=True, drop_artificial=True, integering=True)
    
    
    

    
    # with open(f"{path}/user_lang_dict.json", "w") as f:  ##For bento_service.py
    #     json.dump(user_lang_dict, f)
        
    # with open(f"{path}/pik_lang_dict.json", "w") as f:   ##For bento_service.py
    #     json.dump(pik_lang_dict, f)

    # with open(f"{path}/link_lang_dict.json", "w") as f:   ##For bento_service.py
    #     json.dump(link_lang_dict, f)
    
    # with open(f"{path}/pik_status_dict.json", "w") as f:   ##For bento_service.py
    #     json.dump(pik_status_dict, f)


    with open(f"{path}/user_lang_dict_userset.json", "w") as f:  ##For bento_service.py
        json.dump(user_lang_dict_userset, f)
        
    with open(f"{path}/pik_lang_dict_userset.json", "w") as f:   ##For bento_service.py
        json.dump(pik_lang_dict_userset, f)

    with open(f"{path}/link_lang_dict_userset.json", "w") as f:   ##For bento_service.py
        json.dump(link_lang_dict_userset, f)
    
    with open(f"{path}/pik_status_dict.json", "w") as f:   ##For bento_service.py
        json.dump(pik_status_dict, f)

    with open(f"{path}/linkid_title_dict.json", "w") as f:   ##For bento_service.py
        json.dump(linkid_title_dict, f)
        
    with open(f"{path}/pikid_title_dict.json", "w") as f:   ##For bento_service.py
        json.dump(pikid_title_dict, f)
    
    
    # link_cat_pik.to_csv(f'{path}/link_cat_pik.csv', index=False)
    link_cat_pik.to_csv(f'{path}/link_cat_pik.csv', index=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # with open("/home/hojun/temp/data_development/user_lang_dict.json", "w") as f:  ##For bento_service.py
    #     json.dump(user_lang_dict, f)
        
    # with open("/home/hojun/temp/data_development/pik_lang_dict.json", "w") as f:   ##For bento_service.py
    #     json.dump(pik_lang_dict, f)

    # with open("/home/hojun/temp/data_development/link_lang_dict.json", "w") as f:   ##For bento_service.py
    #     json.dump(link_lang_dict, f)
    
    # with open("/home/hojun/temp/data_development/pik_status_dict.json", "w") as f:   ##For bento_service.py
    #     json.dump(pik_status_dict, f)
    
    
    # # link_cat_pik.to_csv(f'{path}/link_cat_pik.csv', index=False)
    # link_cat_pik_public_private_all_inner.to_csv('/home/hojun/temp/data_development/link_cat_pik_public_private_all_inner.csv', index=False)
    
    
    

    # # from glob import glob
    # slug1 = link_cat_pik['slug'][link_cat_pik['pik_id'] == 3085].iloc[0]
    # slug2 = link_cat_pik['slug'][link_cat_pik['pik_id'] == 3186].iloc[0]
    # slug3 = link_cat_pik['slug'][link_cat_pik['pik_id'] == 3188].iloc[0]
    
    # ##특정 string이 포함된 행을 df에서찾기
    # dummy1 = link_cat_pik[link_cat_pik['slug'].str.contains(slug1)]
    # dummy2 = link_cat_pik[link_cat_pik['slug'].str.contains(slug2)]
    # dummy3 = link_cat_pik[link_cat_pik['slug'].str.contains(slug3)]
    
    # ## 추출한 행을 df에서제거하기
    # link_cat_pik = link_cat_pik[~link_cat_pik.index.isin(dummy1.index)]
    # link_cat_pik = link_cat_pik[~link_cat_pik.index.isin(dummy2.index)]
    # link_cat_pik = link_cat_pik[~link_cat_pik.index.isin(dummy3.index)]
    
    # ##제거된것을확인
    # link_cat_pik[link_cat_pik['slug'] == slug1]
    # link_cat_pik[link_cat_pik['slug'] == slug2]
    # link_cat_pik[link_cat_pik['slug'] == slug3]
    
    # link_cat_pik.drop(['description', 'memo', 'url', 'is_draft_link', 'link_create_time', 'id_cat', 'created_cat', 'id_user', 'id_pik', 'slug', 'language', 'is_draft_catpik', 'created_pik'], axis=1, inplace=True)
    # link_cat_pik = link_cat_pik.astype({'pik_id' :'int', 'link_id':'int', 'category_id':'int'})    
    
    # link_cat_pik = filtering_users(link_cat_pik, 'user_id', artificial_users)  
    
    # link_cat_pik = filter_data(link_cat_pik,  ['link_title'])
    # link_cat_pik = filter_data(link_cat_pik,  ['pik_title'])
    
    # # link_cat_pik.dropna(axis=0, how='any', inplace=True, subset = 'link_title')
    # # link_cat_pik.dropna(axis=0, how='any', inplace=True, subset = 'pik_title')
    # # ti.xcom_push(key='data', value =  link_cat_pik)
    
        

    # user_lang_dict = {str(k):str(v) for k,v in zip(user_language['id'], user_language['language_code'])} ##모든유저를 검색할 수 있도록
    # pik_lang_dict = {k:v for k,v in zip(link_cat_pik['pik_id'], link_cat_pik['language_code'])}
    # link_lang_dict = {k:v for k,v in zip(link_cat_pik['link_id'], link_cat_pik['language_code'])}
    
    
    

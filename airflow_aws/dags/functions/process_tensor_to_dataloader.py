#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:52:02 2022

@author: hojun
"""
from torch.utils.data import DataLoader
import pickle
import torch
from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel, AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd 

def process_sentence(data, tokenizer, tokenizing_col: str, max_len:int=24, return_tensors='pt', padding='max_length', truncation=True):
    # initialize dictionary to store tokenized sentences for link title
    tokens = {'input_ids': [], 'attention_mask': []}
    
    for sentence in tqdm(data[tokenizing_col]):
        # encode each sentence and append to dictionary
        new_tokens = tokenizer.encode_plus(sentence, max_length=24,
                                           truncation=True, padding='max_length',
                                           return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    
    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    return tokens





class PQDataset_torch(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        
    def __getitem__(self, index): 
        return self.input_ids[index], self.attention_masks[index]
        
    def __len__(self): 
        return self.input_ids.shape[0]



def load_model(model_name):
    model = AutoModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
    return model


def load_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)    
    return tokenizer

'''
model_name = 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking'

'''





def process_sent_tensor_to_torchdata(**kwargs):
    '''
    tokenizer_name: name of pretrained tokenizer (see huggingface model)
    processed_data: the processed dataframe file.
    tokenizing_col: column to tokenize sentences of. It can either be link_title or pik_title for now.
    max_len: max_length of tokens to consider for processing. (check encode_plus of huggingface)
    return_tensors: to which data type you want the encoding to be (check encode_plus of huggingface). It can be pt or tf.
    padding: If tokens are shorter than max_len, we choose pad option. (check encode_plus of huggingface)
    batch_size: batch size of dataloader depending on your application and computer specs.
    saving_dataloader_path: dataloader file to be saved. It can be link_title_dataloader or pik_title_dataloader for now.
    '''
    
    #model_name = kwargs.get('model_name', 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking')
    tokenizer_name = kwargs.get('tokenizer_name', 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking')    
    processed_data_path = kwargs.get('processed_data', '/home/ubuntu/airflow/dags/data/link_cat_pik.csv')
    tokenizing_col = kwargs.get('tokenizing_col', 'link_title') ##or pik_title
    max_len = kwargs.get('max_len', 24)
    return_tensors = kwargs.get('return_tensors', 'pt')
    padding = kwargs.get('padding', 'max_length')
    truncation = kwargs.get('truncation', True)
    batch_size = kwargs.get('batch_size', 256)
    saving_dataloader_path = kwargs.get('saving_dataloader_path', f'/home/ubuntu/airflow/dags/data/{tokenizing_col}_dataloader.pickle')
    
    tokenizer = load_tokenizer(tokenizer_name)
    processed_data = pd.read_csv(processed_data_path)
    # processed_data=processed_data.dropna(subset=['link_title', 'pik_title'])
    print(processed_data.shape)
    
    
    tokens = process_sentence(processed_data, tokenizer, tokenizing_col, max_len, return_tensors, padding, truncation)
    dataset = PQDataset_torch(tokens['input_ids'], tokens['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) 
    with open(saving_dataloader_path,'wb') as f:
        pickle.dump(dataloader, f)
    
    
    
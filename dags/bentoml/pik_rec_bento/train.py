#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:53:45 2022

@author: hojun
"""

import torch
from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel, AutoTokenizer, AutoModel, Trainer, TrainingArguments

model_name = 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)#.to('cuda' if torch.cuda.is_available() else 'cpu')


# aa = 'i am a boy'
# z = tokenizer.encode_plus(aa, max_length=24,
#                                     truncation=True, padding='max_length',
#                                     return_tensors='pt')


# def tokenize_function(examples):
#     return tokenizer.encode_plus(sentence, max_length=24, truncation=True, padding='max_length', return_tensors='pt')
    

def tokenize_function(text):
    outputs  = tokenizer.encode_plus(text, max_length=24,
                                        truncation=True, padding='max_length',
                                        return_tensors='pt')#.to('cuda' if torch.cuda.is_available() else 'cpu')
    return outputs
    #return outputs['input_ids'], outputs['attention_mask']



import bentoml
from transformers import pipeline

emb_extractor = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
bentoml.transformers.save_model(name="pik_recommender_model", pipeline=emb_extractor)

# np.array(emb_extractor('i am a boy')[0][0]).shape

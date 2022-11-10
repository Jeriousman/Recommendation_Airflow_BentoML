#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:53:45 2022

@author: hojun
"""

import torch
from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel, AutoTokenizer, AutoModel, Trainer, TrainingArguments
import bentoml
from transformers import pipeline

def load_model_tokenizer(model_name, tokenizer_name):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(model_name)#.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model, tokenizer

def make_bento_model(**kwargs):
    
    model_name = kwargs.get('model_name', 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking')  
    tokenizer_name = kwargs.get('tokenizer_name', 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking')  
    huggingface_pipeline_name = kwargs.get('huggingface_pipeline_name', 'feature-extraction')  
    bentoml_model_name = kwargs.get('bentoml_model_name', 'pik_recommender_model')  
 
    
    model, tokenizer = load_model_tokenizer(model_name, tokenizer_name)
    emb_extractor = pipeline(huggingface_pipeline_name, model=model, tokenizer=tokenizer)
    bentoml.transformers.save_model(name=bentoml_model_name, pipeline=emb_extractor)


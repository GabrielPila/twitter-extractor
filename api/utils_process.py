import pandas as pd
from tqdm import tqdm
import numpy as np

def get_public_metrics(dataset):
    '''Extrae la columna "public_metrics" que se encuentra en el dataset'''
    df_public = pd.DataFrame()
    for i, row in dataset.iterrows():
        public_metrics = eval(row['public_metrics'])
        public_metrics['id'] = row['id']
        df_i = pd.DataFrame([public_metrics])
        df_public = df_public.append(df_i).reset_index(drop=True)        
    return df_public

def get_referenced_tweets(dataset):
    '''Extrae la columna "referenced_tweets" que se encuentra en el dataset'''
    df_referenced = pd.DataFrame()
    for i, row in dataset.iterrows():
        try:
            referenced_tweets = eval(row['referenced_tweets'])[0]
            referenced_tweets['id_ref'] = referenced_tweets['id']
        except:
            referenced_tweets = {}
        referenced_tweets['id'] = row['id']

        df_i = pd.DataFrame([referenced_tweets])
        df_referenced = df_referenced.append(df_i)
    return df_referenced


def get_mentions(text):
    '''Extrae las cuentas de twitter mencionadas'''
    split = text.split('@')
    mentions = [f"@{x.split(' ')[0]}" for x in split if len(x.split(' ')[0]) > 4]
    return mentions


def get_clean_text(text):
    '''Limpia el texto eliminando las menciones'''
    split = text.split('@')
    mentions = [f"@{x.split(' ')[0]}" for x in split if len(x.split(' ')[0]) > 4]
    
    for mention in mentions:
        text = text.replace(mention, '')
    text = text.replace('  ', ' ')
    text = text.replace(' ,', '')
    text = text.replace('\n', '')
    return text


def preprocess_json_cols(dataset):
    '''Realiza todo el procesamiento de limpieza con las funciones antes definidas'''
    dataset_1 = dataset.copy()
    
    public_metrics = get_public_metrics(dataset)
    referenced_tweets = get_referenced_tweets(dataset)
    
    dataset_1 = dataset_1.merge(public_metrics, on='id', how='left')
    dataset_1 = dataset_1.merge(referenced_tweets, on='id', how='left')
    dataset_1 = dataset_1.drop(['public_metrics', 'geo', 'referenced_tweets'], axis=1)
    
    dataset_1['mentions'] = dataset_1['text'].apply(get_mentions)
    dataset_1['num_mentions'] = dataset_1['mentions'].str.len()
    dataset_1['clean_text'] = dataset_1['text'].apply(get_clean_text)
    return dataset_1
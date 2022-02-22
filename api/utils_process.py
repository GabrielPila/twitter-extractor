import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from transformers import pipeline
import dataframe_image as dfi
import seaborn as sns
import emoji
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def get_public_metrics(dataset:pd.DataFrame)->pd.DataFrame:
    '''Extrae la columna "public_metrics" que se encuentra en el dataset'''
    df_public = pd.DataFrame()
    for i, row in dataset.iterrows():
        public_metrics = eval(row['public_metrics'])
        public_metrics['id'] = row['id']
        df_i = pd.DataFrame([public_metrics])
        df_public = df_public.append(df_i).reset_index(drop=True)        
    return df_public

def get_referenced_tweets(dataset:pd.DataFrame)->pd.DataFrame:
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

def get_words_by_key(text:str, key:str)->list:
    '''Obtiene el listado de palabras que empiezan por un key. Útil para colectar links, menciones y hashtags'''
    indexes = [i for i in range(len(text)) if text.startswith(key, i)]
    words = [text[x:].split(' ')[0] for x in indexes]
    return words

def get_clean_text(text:str)->str:
    '''Limpia el texto eliminando las menciones'''
#    hashtags = get_words_by_key(text, '#')
    mentions = get_words_by_key(text, '@')
    links = get_words_by_key(text, 'http')
    emojis = get_emojis(text)
    
    for word in mentions+links+emojis: # No se borrarán hashtags
        text = text.replace(word, '')
    text = clean_vowels(text)
    text = clean_punctuation(text)    
    text = text.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
    text = text.replace(' ,', '')
    text = text.replace('\n', '').strip()
    return text


def get_sentiment_and_prob(text:str, sentiment:pipeline)->float:
    '''Evaluates and predicts the sentiment based on a text'''
    pred = sentiment(text)[0]
    preds = int(pred['label'][0]) + pred['score']
    return preds



def clean_vowels(text:str)->str:
    text = text.lower()
    vowels_l1 = {
        'a':['ä','â','á','à'],
        'e':['ë','ê','é','è'],
        'i':['ï','î','í','ì'],
        'o':['ö','ô','ó','ò'],
        'u':['ü','û','ú','ù']
    }

    vowels_l2 = {}
    for key in vowels_l1:
        for value in vowels_l1[key]:
            vowels_l2[value] = key
    
    for key in vowels_l2:
        text = text.replace(key, vowels_l2[key])
    return text

def clean_punctuation(text:str)->str:
    punctuation = '!"#$%&\()*…+,-./:;<=>?@^_`{|}~¿`~«¨'
    text = re.sub(f'[{punctuation}]', ' ', text)
    return text

def get_emojis(text:str)->list:
    emojis = [x for x in text if x in emoji.UNICODE_EMOJI['en']]
    return emojis

def get_upper_words(text:str)->list:
    emojis = [x for x in text if x in emoji.UNICODE_EMOJI['en']]
    for emj in emojis:
        text = text.replace(emj, '')
    upper_words = [x.replace(',','') for x in text.split() if (len(x)>2) and (x==x.upper())]
    return upper_words

def get_stop_words(text_serie:pd.Series, top_n:int=50, n_len:int=3)->list:
    '''Obtains stopwords from text as the most frequent short words and nltk stopwords'''
    # Detect Stopwords
    full_text = ' '.join(text_serie.to_list())
    full_text = full_text.lower()
    words = pd.Series(full_text.split()).value_counts().head(top_n)
    detected_stop_words = [w for w in words.index if len(w) <= n_len]
    
    # Nltk Stopwords
    stop_words = detected_stop_words + stopwords.words('spanish')
    stop_words = sorted(list(set(stop_words)))
    return stop_words

def get_normalized_text(text:str, stop_words:list=None)->str:
    text = text.lower()
    if stop_words:
        text = ' '.join([x for x in text.split() if x not in stop_words])
    text = text.replace('  ',' ').replace('  ',' ').replace('  ',' ')
    text = text.replace('  ',' ').replace('  ',' ').replace('  ',' ')
    return text


def process_dataset(sentiment, dataset:pd.DataFrame, print_progress:bool=False) -> pd.DataFrame:
    '''Realiza todo el procesamiento de limpieza con las funciones antes definidas'''
    start_time = time.time()
    dataset_1 = dataset.copy()
    
    # Obtaining Json Data
    public_metrics = get_public_metrics(dataset)
    referenced_tweets = get_referenced_tweets(dataset)
    
    # Merging Json Data
    dataset_1 = dataset_1.merge(public_metrics, on='id', how='left')
    dataset_1 = dataset_1.merge(referenced_tweets, on='id', how='left')
    dataset_1 = dataset_1.drop(['public_metrics', 'geo', 'referenced_tweets'], axis=1)
    if print_progress:
        print('INFO: Jsons procesed and merged')
    
    # Cleaning Data Elements
    dataset_1['mentions'] = dataset_1['text'].apply(get_words_by_key, args=['@'])
    dataset_1['hashtags'] = dataset_1['text'].apply(get_words_by_key, args=['#'])
    dataset_1['links'] = dataset_1['text'].apply(get_words_by_key, args=['http'])
    dataset_1['emojis'] = dataset_1['text'].apply(get_emojis)
    dataset_1['upper_words'] = dataset_1['text'].apply(get_upper_words)
    
    dataset_1['num_mentions'] = dataset_1['mentions'].str.len()
    dataset_1['num_hashtags'] = dataset_1['hashtags'].str.len()
    dataset_1['num_links'] = dataset_1['links'].str.len()
    dataset_1['num_emojis'] = dataset_1['emojis'].str.len()
    dataset_1['num_upper_words'] = dataset_1['upper_words'].str.len()
    
    dataset_1['date'] = pd.to_datetime(dataset_1['created_at'].str[:10])
    
    if print_progress:
        print('INFO: List of elements processed')
    
    dataset_1['clean_text'] = dataset_1['text'].apply(get_clean_text)
    stop_words = get_stop_words(dataset_1['clean_text'])
    dataset_1['norm_text'] = dataset_1['clean_text'].apply(get_normalized_text, args=[stop_words])
    if print_progress:
        print('INFO: Text cleaned')
    
    # Process Sentiment Data
    if print_progress:
        print('INFO: Processing Sentiment')
    tqdm.pandas()
    dataset_1['sentiment_output'] = dataset_1['clean_text'].progress_apply(get_sentiment_and_prob, args=[sentiment])
    dataset_1['sentiment_pred'] = (dataset_1['sentiment_output']//1).astype(int)
    dataset_1['sentiment_conf'] = dataset_1['sentiment_output']%1
    dataset_1['sentiment_pred_conf45'] = np.nan
    dataset_1.loc[dataset_1['sentiment_conf']>0.45, 'sentiment_pred_conf45']=dataset_1['sentiment_pred']
    if print_progress:
        print('INFO: Sentiment Obtained')
    
    processing_time = (time.time() - start_time)
    
    if print_progress:
        print(f'INFO: Processing Time:\t{processing_time:.2f} seconds')

    return dataset_1



def get_emoji_table(emoji_serie:pd.Series, top_n:int=10):
    emoji_serie_joined = emoji_serie.apply(lambda x: ' '.join(x))
    emoji_list = ' '.join(emoji_serie_joined.to_list())
    emoji_counter = pd.Series(emoji_list.split()).value_counts().head(top_n)
    emoji_counter = emoji_counter.to_frame('ocurrencias').T
    return emoji_counter

def generate_wordcloud(text_serie:pd.Series,
                       extra_title:str='', 
                       figsize:tuple=(15,6),
                       fontsize:int=26,
                       color='#347FC4',
                       fontname='Helvetica'
                      ):
    '''Plots a WordCloud based on a pd serie of Texts.
    The text should preferably be cleaned and normalized.'''
    
    full_text = ' '.join(text_serie.to_list())

    wordcloud = WordCloud(background_color="white", max_font_size=50, max_words=50)
    wordcloud = wordcloud.generate(full_text)

#    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    title = f'WordCloud {extra_title}'
    plt.title(title, fontsize=fontsize, color=color, fontname=fontname)
#    plt.show()

def plot_from_list_serie(list_serie:pd.Series, 
                         n:int=10, 
                         figsize:tuple=(8,3),
                         color:str='orange',
                         fontname:str='helvetica',
                         fontsize:int=16
                        ):
    serie = list_serie.apply(lambda x: ' '.join(x))
    list_from_serie = ' '.join(serie.to_list())
    serie_counted = pd.Series(list_from_serie.split()).value_counts()
#    fig = plt.figure()
    serie_counted = serie_counted.head(n).sort_values()
    plt.barh(y=serie_counted.index, width=serie_counted.values, color=color)
    plt.title(f'Top {n} {serie.name}', color=color, fontname=fontname, fontsize=fontsize)
    plt.axvline(0)
    sns.despine()
#    plt.axis('x')

def plot_barchart(serie, 
                  title = 'Cantidad de Tweets', 
                  fontsize=26,
                  fontname='Helvetica',
                  color='Orange',
                  rotation=90
                 ):
    plt.bar(x=serie.index, height=serie.values)
    plt.title(title, fontsize=fontsize, fontname=fontname, color=color)
    sns.despine()
    plt.xticks(rotation=90);



def plot_report(
    # PLOT FINAL
    dataset,
    figsize=(16,30), 
    fontsize=26,
    color='#347FC4',
    fontname='Helvetica',
    nrows=7,
    ncols=2,
    i=1
):
    keyword = dataset['keyword'].values[0].split('-')[0].strip()

    PATH_REPORT = 'reporte'
    if not os.path.exists(PATH_REPORT):
        os.mkdir(PATH_REPORT)

    # Export de Emojis
    report = get_emoji_table(dataset['emojis'])
    dfi.export(report, f'{PATH_REPORT}/{keyword}_emojis.png')
    img = plt.imread(f'{PATH_REPORT}/{keyword}_emojis.png')

    
    fig = plt.figure(figsize=figsize)
    
    fig.add_subplot(nrows,ncols,i)
    first_title = f'Reporte de Data extraída\nde Twitter\n\nKeyword - {keyword}'
    plt.text(x=0.5, y=0.5, s=first_title, 
             fontdict={'fontsize':'40',
                       'fontweight':'bold',
                       'ha':'center',
                       'va':'center',
                       'color':color,
                       'fontname':fontname})

    plt.axis('off')


    i+=1
    fig.add_subplot(nrows,ncols,i)
    # Conteo de Sentimiento
    serie = dataset['sentiment_pred'].value_counts().sort_index()
    colors = ['#7bdff2', '#b2f7ef', '#eff7f6', '#f7d6e0', '#f2b5d4']
    plt.pie(serie, labels=serie.index, 
            autopct='%.1f%%', 
            pctdistance=0.75,
            colors=colors,
            labeldistance=1.03,
            startangle=90,
            textprops={'size': 'large'},
            wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'}
            )
    plt.text(1,0.8,'Multilingual Sent\n1: Bad\n5: Good')
    plt.title('Distribución de Sentimiento', fontname=fontname, fontsize=fontsize, color=color)



    i+=1
    fig.add_subplot(nrows,ncols,i)
    # Conteo de Tweets
    serie = dataset.groupby('date').agg({'text':'count'})['text']
    title = 'Cantidad de Tweets'
    plot_barchart(serie, title='Cantidad de Tweets', 
                  fontsize=fontsize, fontname=fontname, color=color)


    i+=1
    fig.add_subplot(nrows,ncols,i)
    # WordCloud Total Texto Limpio y Normalizado
    generate_wordcloud(dataset['norm_text'], extra_title='Tweets', fontsize=fontsize, color=color, fontname=fontname)


    i+=1
    fig.add_subplot(nrows,ncols,i)
    # Sentiment Promedio en el tiempo
    serie = dataset.groupby('date').agg({'sentiment_pred_conf45':'mean'})['sentiment_pred_conf45']
    title = 'Sentiment Promedio'
    plt.plot(serie, marker='o', linestyle='-.', linewidth=0.5, color=color, mec='r', mfc='r')
    plt.title(title, fontname=fontname, fontsize=fontsize, color=color)
    sns.despine()
    plt.xticks(rotation=90);


    i+=1
    fig.add_subplot(nrows,ncols,i)
    # Top Emojis
    plt.imshow(img)
    plt.title('Top Emojis en Tweets', fontsize=fontsize, color=color, fontname=fontname)
    plt.axis('off')


    i+=1
    fig.add_subplot(nrows,ncols,i)
    # Top Mentions
    plot_from_list_serie(dataset['mentions'], fontsize=fontsize, color=color, fontname=fontname)


    i+=1
    fig.add_subplot(nrows,ncols,i)
    # WordCloud Mentions
    generate_wordcloud(dataset['mentions'].apply(lambda x: ' '.join(x)), 
                       extra_title='Mentions', fontsize=fontsize, color=color, fontname=fontname)


    i+=1
    fig.add_subplot(nrows,ncols,i)
    # Top Hashtags
    plot_from_list_serie(dataset['hashtags'], fontsize=fontsize, color=color, fontname=fontname)


    i+=1
    fig.add_subplot(nrows,ncols,i)
    # WordCloud Hashtags
    generate_wordcloud(dataset['hashtags'].apply(lambda x: ' '.join(x)), 
                       extra_title='Hashtags', fontsize=fontsize, color=color, fontname=fontname)


    i+=1
    fig.add_subplot(nrows,ncols,i)
    # Top Mayusculos
    plot_from_list_serie(dataset['upper_words'], fontsize=fontsize, color=color, fontname=fontname)


    i+=1
    fig.add_subplot(nrows,ncols,i)
    # WordCloud Mayusculos
    generate_wordcloud(dataset['upper_words'].apply(lambda x: ' '.join(x)), 
                       extra_title='Mayusculas', fontsize=fontsize, color=color, fontname=fontname)


    plt.tight_layout();

    fig.savefig(f'{PATH_REPORT}/Reporte_{keyword}.pdf');
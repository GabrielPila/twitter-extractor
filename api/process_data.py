import pandas as pd
import os
import time
from transformers import pipeline
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import nltk 
from nltk.corpus import stopwords
import emoji
import numpy as np
from tqdm import tqdm
import seaborn as sns
import dataframe_image as dfi
import warnings
from utils_process import process_dataset, plot_report

pd.set_option('display.max_columns', None)
pd.options.display.max_colwidth = 200
warnings.filterwarnings('ignore')

PATH_CONSOLIDADO = 'data_consolidated/'
files = sorted(os.listdir(PATH_CONSOLIDADO))
files = [x for x in files if 'data' in x]

data = {}
for file in files:
    data[file] = pd.read_csv(os.path.join(PATH_CONSOLIDADO, file))

# Initialize Sentiment Model
model_multilingual_path = 'nlptown/bert-base-multilingual-uncased-sentiment'
sentiment = pipeline(task = 'sentiment-analysis',
                     model = model_multilingual_path,
                     tokenizer = model_multilingual_path)

for file in files:
    print(file)
    dataset = data[file]
    dataset = process_dataset(sentiment, dataset, print_progress=False)
    plot_report(dataset)
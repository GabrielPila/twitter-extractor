import requests
import os
import json
import pandas as pd
import datetime
from tqdm import tqdm

from settings import BEARER_TOKEN
from params import search_url, keyword, start_time, max_results, step_time, num_steps
from utils import create_headers, create_url, connect_to_endpoint


#Inputs for the request
bearer_token = BEARER_TOKEN
headers = create_headers(bearer_token)

start_time = pd.to_datetime(start_time)

if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists('data_users'):
    os.mkdir('data_users')

for i in tqdm(list(range(num_steps))):
    start = str(start_time + datetime.timedelta(minutes = i * step_time))
    end = str(start_time + datetime.timedelta(minutes = (i+1) * step_time))

    start = start.replace(" ", "T").replace("+00.00",".000Z")
    end = end.replace(" ", "T").replace("+00.00",".000Z")

    params = create_url(keyword, start, end, max_results)
    json_response = connect_to_endpoint(search_url, headers, params)
    timestamp = str(datetime.datetime.now())[:-4]
    end_file = f'{keyword.split(" ")[0]}_start_{start.split("+")[0]}_end_{end.split("+")[0]}'

    data = pd.DataFrame(json_response['data'])
    data['keyword'] = keyword
    data['timestamp'] = timestamp
    data.to_csv(f'data/dataTW_GP_{end_file}.csv', index=False)

    users = pd.DataFrame(json_response['includes']['users'])
    users['keyword'] = keyword
    users['timestamp'] = timestamp
    users.to_csv(f'data_users/usersTW_GP_{end_file}.csv', index=False)


#params = create_url(keyword, start_time, end_time, max_results)
#json_response = connect_to_endpoint(search_url, headers, params)
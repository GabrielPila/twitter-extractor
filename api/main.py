import requests
import os
import json
import pandas as pd
import csv
import datetime
import dateutil.parser
import unicodedata
import time

from settings import BEARER_TOKEN
from params import search_url, keyword, start_time, end_time, max_results
from utils import create_headers, create_url, connect_to_endpoint


#Inputs for the request
bearer_token = BEARER_TOKEN
headers = create_headers(bearer_token)


params = create_url(keyword, start_time,end_time, max_results)
json_response = connect_to_endpoint(search_url, headers, params)

print(list(json_response))
# print(json.dumps(json_response, indent=4, sort_keys=True))
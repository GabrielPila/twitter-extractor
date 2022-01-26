search_url = "https://api.twitter.com/2/tweets/search/recent" #Change to the endpoint you want to collect data from
keyword = "@JaimeChincha -is:retweet lang:es" 
start_time = "2022-01-24T08:00:00.000Z" # start of the extraction
#end_time = "2022-01-23T00:00:00.000Z"
step_time = 10 # minutes
num_steps = 60 # number of extractions
max_results = 100 # number of results extracted per request
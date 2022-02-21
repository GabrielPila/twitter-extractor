search_url = "https://api.twitter.com/2/tweets/search/recent" #Change to the endpoint you want to collect data from
keyword = "Malcricarmen -is:retweet lang:es" 
start_time = "2022-02-20T20:10:00.000Z" # start of the extraction
#start_time = "2022-02-10T09:00:00.000Z" # start of the extraction
#end_time = "2022-01-23T00:00:00.000Z"
step_time = 10 # minutes
num_steps = 720 # number of extractions
max_results = 100 # number of results extracted per request
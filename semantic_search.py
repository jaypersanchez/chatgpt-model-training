import pandas as pd
import numpy as np
import openai 
import os
# Import psycopg2 for database connectivity - Postgresql
import psycopg2
import json
import csv

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
# Token Tuning 
token_tuning_params = {
    "num_tokens": 4, # the number of tokens to use
    "max_tokens": 10, # the maximum number of tokens to generate
    "min_token_length": 4, # the minimum length of each token
    "max_token_length": 8 # the maximum length of each token
}

# Initialize the OpenAI class
openai.organization = os.environ['OPENAI_ORG_KEY']
openai.api_key = os.environ['OPENAI_API_KEY']

#check if we are authenticated
modelList = openai.Engine.list()
#print out the available data model for embedding
#print(modelList)

# load the csv file 
csv_data = [] 
#need to cleanup data
with open('./models/exchange.csv', 'r', encoding='utf-8') as csvfile: 
    reader = csv.reader(csvfile) 
    #for row in reader: 
        #csv_data.append(row) 
    for i, row in enumerate(reader):
        if i < 100:
            csv_data.append(reader)
    i += 1
            
# convert the csv data into a json format 
json_data = json.dumps(csv_data)
res = openai.Embedding.create(data=json_data, model=embedding_model,input=json_data, params=max_tokens)
print(res)
embeddeing_list = [item["embedding"] for item in res["data"]]
print(embeddeing_list)

# Establish a connection to the PostgreSQL database
conn = psycopg2.connect("dbname=Satoshi user=postgres password=Ph1LL!fe host=localhost")

# Create a cursor object
cur = conn.cursor()

#vectors = res[input]

#print out the vector value of the data aboveprint(res)
#get all the vector values from res.data[0].embedding
#for i in range(len(vectors)):
#     cur.execute("INSERT INTO vectors (text, vector) VALUES (%s, %s)", (res["input"][i], vectors[i]))
     # Commit the changes

#save into vectors table
#conn.commit()
     
# Close the connection
conn.close()

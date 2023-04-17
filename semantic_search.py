import pandas as pd
import numpy as np
import openai 
import os
# Import psycopg2 for database connectivity - Postgresql
import psycopg2
import json
import csv
# Initialize the OpenAI class
openai.organization = os.environ['OPENAI_ORG_KEY']
openai.api_key = os.environ['OPENAI_API_KEY']

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 3000  # the maximum for text-embedding-ada-002 is 8191
conn = psycopg2.connect("dbname=Satoshi user=postgres password=Ph1LL!fe host=localhost")
cursor = conn.cursor()

#Get text prompt
userPrompt = "what are the symbols starting with AAA"
#convert userPrompt into vector value
res = openai.Embedding.create(
    input=[
        userPrompt
    ],engine=embedding_model
)
embedding_list = [item["embedding"] for item in res["data"]]
#print(embedding_list)


# Get the top 10 most similar vectors based on cosine similarity
query_vector = [embedding_list]  # example query vector
cursor.execute("SELECT csv_data, embedding_list, similarity(embedding_list, %s) FROM vectors ORDER BY similarity", (query_vector, ))

results = cursor.fetchall()
print(results)

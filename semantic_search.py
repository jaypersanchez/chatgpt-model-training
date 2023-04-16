import pandas as pd
import numpy as np
import openai 
import os
# Import psycopg2 for database connectivity - Postgresql
import psycopg2

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# Initialize the OpenAI class
openai.organization = "org-PwafGfC5oVQgjaAzYFmgp1ep"
openai.api_key = "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"
#check if we are authenticated
modelList = openai.Engine.list()
#print out the available data model for embedding
#print(modelList)

res = openai.Embedding.create(
    input = [
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine = embedding_model
)
#print(res)
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

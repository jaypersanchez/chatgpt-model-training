import pandas as pd
import numpy as np
import openai 
import os
# Import psycopg2 for database connectivity - Postgresql
import psycopg2
import json
import csv
from openai.embeddings_utils import get_embedding, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

#open csv file and read it
def prepareData():
  with open('./models/exchange.csv', 'r', newline="", encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) #skip the header
    for row in reader:
      #combine columns into single column
      new_text_data = ', '.join(row[1:])
      #remove commas
      new_text_data = new_text_data.replace(',', '')
      #fix any empty columns
      if new_text_data == '':
        new_text_data = 'N/A' 
      #append the new_text_data to the cleaned_data.csv
      with open('./models/cleaned_data.csv', 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([new_text_data])
        
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.strip()
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def create_embedding():
  with open('./models/cleaned_data.csv', 'r') as f:
    reader = csv.reader(f)
    with open('./models/exchange_embeddings.csv', 'w', newline='') as f:
      writer = csv.writer(f)
      #writer.writerow(['Text', 'Embedding'])
      for row in reader:
          text = row[0]
          embedding = get_embedding(text)
          writer.writerow([text, embedding])
        
#prepareData()
embedding_vector = create_embedding()
#semantic_search("CANADIAN SECURITIES")

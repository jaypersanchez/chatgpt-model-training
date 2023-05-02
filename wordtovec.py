import pandas as pd
import gensim 
from gensim.models import Word2Vec

# Read the CSV
df = pd.read_csv('./models/data.csv')

# Create the list of texts
texts = [text.split() for text in df['text']]

# Train the model
model = Word2Vec(texts, size=100, window=5, min_count=1, workers=4)

# Perform semantic search using the model
semantic_search = model.most_similar('XCNQ')

# Print the results
print(semantic_search)
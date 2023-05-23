# Import the necessary libraries
from gensim.models import TfidfModel
import numpy as np

# Create a list of documents
documents = [
    "I love watching movies.",
    "The best pizza is in Italy.",
    "I enjoy swimming in the ocean.",
    "Machine learning is fascinating."
]

# Create a TfidfModel
model = TfidfModel(documents)

# Create a vector representation of the query
query_vector = np.mean([model[word] for word in "what do I love to do".split() if word in model], axis=0)

# Calculate the similarity score between the query and each document
similarities = [np.dot(query_vector, model[word]) for word in model]

# Sort the documents by their similarity score
sorted_documents = [document for score,document in sorted(zip(similarities,documents), reverse=True)]

# Print the most similar documents
print(sorted_documents[:3])
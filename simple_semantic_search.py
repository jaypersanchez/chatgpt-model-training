import openai
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
openai_api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = openai.api_key = openai_api_key
model_engine = "text-embedding-ada-002"
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


documents = [
    "I love watching movies.",
    "The best pizza is in Italy.",
    "I enjoy swimming in the ocean.",
    "Machine learning is fascinating.",
]

def generate_embeddings(texts):
    prompt = "Create an embedding for the following texts:"
    completions = []

    for text in texts:
        prompt += f"\n- {text}"

    result = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=len(texts) * 16,  # 16 tokens per embedding
        n=1,
        stop=None,
        temperature=0.5,
    )
    completions = result.choices[0].text.strip().split("\n")

    embeddings = [np.fromstring(comp.strip()[1:-1], sep=",") for comp in completions]
    return np.array(embeddings)


def semantic_search(query, embeddings, top_k=1):
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [(index, similarities[index]) for index in top_indices]



# Call to generate embeddings from the documents array
#document_embeddings = generate_embeddings(documents)
document_embeddings = model.encode(documents)

#perform a semantic search
#query = "What are your thoughts about artificial intelligence?"
query = "I'm bored what should I do?"
results = semantic_search(query, document_embeddings)

for index, similarity in results:
    print(f"Document: {documents[index]} - Similarity: {similarity:.4f}")





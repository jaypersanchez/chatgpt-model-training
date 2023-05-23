import openai
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
openai_api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = openai.api_key = openai_api_key
#model_engine = "text-embedding-ada-002"
model_engine = "gpt-4"
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


documents = [
    "I love watching movies.",
    "The best pizza is in Italy.",
    "I enjoy swimming in the ocean.",
    "Machine learning is fascinating.",
    "GB  EURONEXT    UK  -  REPORTING  SERVICES  DRSP  DRSP  WWW.EURONEXT.COM",
    "CA  CANADIAN  SECURITIES  EXCHANGE  XCNQ  XCNQ  WWW.THECSE.COM",
    "CA  CANADIAN  SECURITIES  EXCHANGE  -  PURE  PURE  XCNQ  WWW.THECSE.COM",
    "GB  ZODIA  MARKETS  ZODM  ZODM  WWW.ZODIA-MARKETS.COM",
    "US  FENICS  FX  ECN  FNFX  BGCF  WWW.FENICSFX.COM",
    "NO  NASDAQ  OSLO  ASA  NORX  NORX  WWW.NASDAQ.COM/SOLUTIONS/EUROPEAN-COMMODITIES",
    "ES  PORTFOLIO  STOCK  EXCHANGE  POSE  POSE  WWW.PORTFOLIO.EXCHANGE",
    "US  PUNDION  LLC  PUND  PUND  WWW.PUNDION.COM",
    "BG  UNICREDIT  BULBANK  AD  -  SYSTEMATIC  INTERNALISER  UCBG  UCBG  WWW.UNICREDITBULBANK.BG",
    "AU  ASX  -  NEW  ZEALAND  FUTURES  &  OPTIONS  NZFX  XASX  WWW.ASX.COM.AU",
    "US  ONECHICAGO  LLC  XOCH  XOCH  WWW.ONECHICAGO.COM",
    "SG  BONDBLOX  EXCHANGE  BBLX  BBLX  WWW.BONDBLOX.COM"
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
    print(completions)
    embeddings = [np.fromstring(comp.strip()[1:-1], sep=",") for comp in completions]
    print(embeddings)
    return np.array(embeddings)


def semantic_search(query, embeddings, top_k=1):
    query_embedding = model.encode([query])[0]
    #print(query_embedding)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    #print(similarities)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    #print(top_indices)
    return [(index, similarities[index]) for index in top_indices]



# Call to generate embeddings from the documents array
#document_embeddings = generate_embeddings(documents)
document_embeddings = model.encode(documents)
#print(document_embeddings)
#perform a semantic search
#query = "What are your thoughts about artificial intelligence?"
query = "what are in Canada?"
results = semantic_search(query, document_embeddings)
#print(results)
for index, similarity in results:
    print(index, similarity)
    print(f"Document: {documents[index]} - Similarity: {similarity:.4f}")





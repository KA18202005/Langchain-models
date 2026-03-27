from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()   

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)

documents = [
    "Delhi is the capital of India",
    "Mumbai is the financial capital of India",
    "Bangalore is the IT hub of India",
]

query = "What is the capital of India?"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
similarity_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
most_similar_doc_index = np.argmax(similarity_scores)
 
print("Most similar document:", documents[most_similar_doc_index])
print("Similarity score:", similarity_scores[most_similar_doc_index])
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Qdrant

#Initialize Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333")

collection_name = "basic"
vectors_config = VectorParams(size=384, distance="Cosine")  

if qdrant_client.collection_exists(collection_name=collection_name):
    qdrant_client.delete_collection(collection_name=collection_name)

qdrant_client.create_collection(collection_name=collection_name, vectors_config=vectors_config)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample data
texts = ["Hello, how are you?", 
         "What is the weather today?", 
         "Tell me a joke.", 
         "How was the new film?"]

vectors = model.encode(texts)

points = [
    PointStruct(id=i, vector=vector.tolist(), payload={"text": text})
    for i, (vector, text) in enumerate(zip(vectors, texts))
]

qdrant_client.upsert(collection_name=collection_name, points=points)

def query_qdrant(query_text, top_k=2):
    query_vector = model.encode([query_text])[0].tolist()
    results = qdrant_client.search(collection_name=collection_name, query_vector=query_vector, limit=top_k)
    return results

query_text = input("Text: ")
results = query_qdrant(query_text)

for result in results:
    print(f"Score: {result.score}, Text: {result.payload['text']}")
#
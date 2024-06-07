import configparser
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Load the configuration
config = configparser.ConfigParser()
config.read('config.ini')

qdrant_url = config['qdrant']['url']
data_path = config['paths']['data_path']

qdrant_client = QdrantClient(url=qdrant_url)

collection_name = "basic"
vectors_config = VectorParams(size=384, distance="Cosine")

if qdrant_client.collection_exists(collection_name=collection_name):
    qdrant_client.delete_collection(collection_name=collection_name)
qdrant_client.create_collection(collection_name=collection_name, vectors_config=vectors_config)

model = SentenceTransformer('all-MiniLM-L6-v2')

with open(data_path, 'r') as file:
    texts = file.readlines()

vectors = model.encode(texts)

points = [PointStruct(id=i, vector=vector.tolist(), payload={"text": text}) for i, (vector, text) in enumerate(zip(vectors, texts))]
qdrant_client.upsert(collection_name=collection_name, points=points)
print("Done")

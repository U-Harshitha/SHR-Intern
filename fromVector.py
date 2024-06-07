import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Connect to local Qdrant server running in Docker
client = QdrantClient(host="127.0.0.1", port=6333)

# Create a collection if it does not exist
collection_name = "my_collection"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=5,  # Vector size (change according to your vector dimensions)
        distance=models.Distance.COSINE
    )
)

# Function to load vectors from files
def load_vectors_from_files(directory):
    vectors = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            vector = np.loadtxt(file_path, delimiter=',')
            vectors.extend(vector)
    return vectors

# Directory containing the vector files
directory = './Vector-Files'

# Load vectors
vectors = load_vectors_from_files(directory)

# Insert vectors into Qdrant
for idx, vector in enumerate(vectors):
    client.upsert(
        collection_name=collection_name,
        points=[models.PointStruct(id=idx, vector=vector.tolist())]
    )

# Perform a search
search_vector = np.random.rand(5).tolist()  # Replace with your search vector
search_result = client.search(
    collection_name=collection_name,
    query_vector=search_vector,
    limit=5  # Number of results to return
)

print("Search results:")
for result in search_result:
    print(f"ID: {result.id}, Score: {result.score}")

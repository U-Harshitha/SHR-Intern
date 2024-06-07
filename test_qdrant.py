import unittest
import configparser
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

class TestQdrant(unittest.TestCase):
    def setUp(self):
        # Load the configuration
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Get Qdrant URL from the configuration
        qdrant_url = config['qdrant']['url']

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url)

        # Create collection configuration
        self.collection_name = "basic"
      #  vectors_config = VectorParams(size=384, distance="Cosine")

        # Create collection if it doesn't exist
       # if self.qdrant_client.collection_exists(collection_name=self.collection_name):
        #    self.qdrant_client.delete_collection(collection_name=self.collection_name)
        #self.qdrant_client.create_collection(collection_name=self.collection_name, vectors_config=vectors_config)

        # Load model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def test_insert_and_query(self):
        # Sample data
        texts = ["Hello, how are you?", "What is the weather today?"]
        vectors = self.model.encode(texts)

        # Insert data into Qdrant
        points = [PointStruct(id=i, vector=vector.tolist(), payload={"text": text}) for i, (vector, text) in enumerate(zip(vectors, texts))]
        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

        # Query Qdrant for similar vectors
        query_text = "Hello"
        query_vector = self.model.encode([query_text])[0].tolist()
        results = self.qdrant_client.search(collection_name=self.collection_name, query_vector=query_vector, limit=1)

        # Assert that the query result matches the expected text
        self.assertEqual(results[0].payload["text"], "Hello, how are you?")

if __name__ == '__main__':
    unittest.main()

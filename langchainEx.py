from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os

OPENAI_API_KEY ='env.openapikey'

loader = TextLoader("./temporary.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="my_documents",
)

query = "What are plans for education"
found_docs = qdrant.similarity_search(query)

if found_docs:
    print(found_docs[0].page_content)
else:
    print("No relevant documents found.")

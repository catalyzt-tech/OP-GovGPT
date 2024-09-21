import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Load environment variables
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "GCBOT"
COLLECTION_NAME = "GCBOT"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "GCBOT"

EMBEDDING_FIELD_NAME = "embedding"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Document loading and splitting
loader = DirectoryLoader(
    "/Users/suchanatratanarueangrong/Project/LLM/Optimism/AllData",
    glob="**/*.txt",
    use_multithreading=True,
    loader_cls=TextLoader,
)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
docs = text_splitter.split_documents(data)

# Initialize Hugging Face Embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    multi_process=False,  # Disable multiprocessing
)

# Insert the documents into MongoDB Atlas Vector Search
x = MongoDBAtlasVectorSearch.from_documents(
    documents=docs,
    embedding=hf,
    collection=collection,
    index_name="embedding",
)

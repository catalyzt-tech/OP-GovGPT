import os
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereEmbeddings
from pymongo import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from dotenv import load_dotenv

load_dotenv()


class Citation:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=os.environ["GROQ_API_KEY"],
            model_name="llama-3.1-70b-versatile",
            temperature=0,
            max_tokens=512,
        )
        # COHERE
        self.embeddings = CohereEmbeddings(
            model="embed-english-light-v3.0",
            cohere_api_key=os.environ["COHERE_API_KEY"],
        )
        self.MONGO_URI = os.getenv("MONGO_URI")
        self.DB_NAME = "GCBOT"
        self.COLLECTION_NAME = "GCBOT"
        self.ATLAS_VECTOR_SEARCH_INDEX_NAME = "GCBOT"
        self.client = MongoClient(self.MONGO_URI)
        self.db = self.client[self.DB_NAME]
        self.collection = self.db[self.COLLECTION_NAME]
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            index_name=self.ATLAS_VECTOR_SEARCH_INDEX_NAME,
            embedding=self.embeddings,  # Your HuggingFace embedding model
            text_key="text",  # Make sure 'text' is the correct field name in your collection
            embedding_key="embedding",  # Make sure 'embedding' is the correct field name
            filename_key="source",  # Ensure 'source' is the correct field name for the filenames
            relevance_score_fn="cosine",  # Ensure 'cosine' is correctly implemented as a similarity function
        )

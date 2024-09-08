import os
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient


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

    def process_llm_response(self, llm_response):
        true_temp = []
        for filename in llm_response:
            url = filename.replace("_", "/").replace("+", ":").replace(".txt", "")
            if url not in true_temp:
                true_temp.append(url)

        return true_temp


class HybridSearcher:
    DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

    def __init__(self, collection_name):
        self.collection_name = collection_name
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(
            api_key=os.getenv("QDRANT_API_KEY"), location=os.getenv("QDRANT_URL_KEY")
        )
        self.qdrant_client.set_model(self.DENSE_MODEL)
        self.qdrant_client.set_sparse_model(self.SPARSE_MODEL)

    def search(self, text: str):
        search_result = self.qdrant_client.query(
            collection_name=self.collection_name,
            query_text=text,
            query_filter=None,  # If you don't want any filters for now
            limit=7,  # 5 the closest results
        )
        # `search_result` contains found vector ids with similarity scores
        # along with the stored payload

        # Select and return metadata
        metadata = [hit.metadata for hit in search_result]
        return metadata


if __name__ == "__main__":
    searcher = HybridSearcher("startupschunk2")
    result = searcher.search("What is op")
    print(result)

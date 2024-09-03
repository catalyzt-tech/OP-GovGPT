from pymongo import MongoClient
import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_cohere import CohereEmbeddings
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
        # MONGODB
        self.client = MongoClient(os.environ["MONGODB_API_KEY"])
        self.db = self.client["Vector-store"]
        self.collection = self.db["store-2"]
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            text_key="content",
            embedding_key="embedding",
            filename_key="filename",
        ).as_retriever(search_kwargs={"k": 5})

    def qa_chain(self):
        from langchain.chains import RetrievalQA

        try:
            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store,
                return_source_documents=True,
            )
            print("QA Chain initialized successfully.")
            return chain
        except Exception as e:
            print(f"Error initializing QA Chain: {e}")
            raise

    def process_llm_response(self, llm_response):
        print("\nSources:")
        source_temp = []
        true_temp = []
        for source in llm_response.get("source_documents", []):
            filename = source.metadata.get("filename", "")
            if filename not in source_temp:
                source_temp.append(filename)

        for filename in source_temp:
            # Replace underscores with slashes and plus signs with colons, and remove file extension
            url = filename.replace("_", "/").replace("+", ":").replace(".txt", "")
            true_temp.append(url)
            print(url)  # Print the formatted URL

        return true_temp

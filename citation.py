from pymongo import MongoClient
import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch


class Citation:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=os.environ["groq_api"],
            model_name="llama-3.1-70b-versatile",
            temperature=0,
            max_tokens=512,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        self.client = MongoClient(os.environ["MongoDB"])
        self.db = self.client["Vector-store"]
        self.collection = self.db["store-1"]
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            text_key="content",
            embedding_key="embedding",
            filename_key="filename",
        )

    def qa_chain(self):
        from langchain.chains import RetrievalQA

        try:
            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
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

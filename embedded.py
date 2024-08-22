from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

loader = DirectoryLoader("AllData", glob="./*.txt", loader_cls=TextLoader)

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)
persist_directory = "dbAlldata"
vectordb = Chroma.from_documents(
    documents=texts, embedding=embeddings, persist_directory=persist_directory
)

vectordb.persist()
vectordb = None

# retriever = vectordb.as_retriever()
print("Success")

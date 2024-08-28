import os

# Directory where your text files are stored
directory_path = "AllData"

# Load all text files
documents = []
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        with open(
            os.path.join(directory_path, filename), "r", encoding="utf-8"
        ) as file:
            content = file.read()
            documents.append({"filename": filename, "content": content})

# print("Loaded documents:", documents)
from transformers import AutoTokenizer, AutoModel
import torch

# Load a Hugging Face model and tokenizer
model_name = (
    "sentence-transformers/all-MiniLM-L6-v2"  # You can choose a different model
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def embed_text(text):
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get the embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling of the embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()

    return embeddings.tolist()


# Add embeddings to each document using the Hugging Face model
for doc in documents:
    doc["embedding"] = embed_text(doc["content"])

from pymongo import MongoClient

# Connect to MongoDB Atlas
client = MongoClient(os.environ["MONGODB_API_KEY"])
db = client["Vector-store"]
collection = db["store-1"]

# Insert documents into the collection
collection.insert_many(documents)
print("Documents inserted successfully")

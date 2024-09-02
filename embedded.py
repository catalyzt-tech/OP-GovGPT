import os
from transformers import AutoTokenizer, AutoModel
import torch
from pymongo import MongoClient

# Directory where your text files are stored
directory_path = "AllData"

# Load all text files
documents = []
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
            content = file.read()
            documents.append({"filename": filename, "content": content})

# Load a Hugging Face model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # You can choose a different model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def embed_text(text):
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move the inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling of the embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()

    return embeddings.cpu().tolist()

def chunk_text(text, max_length=200, overlap=3):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        # Ensure chunk is within model's token limit
        if len(chunk) > 512:
            chunk = chunk[:512]
        chunk_text = tokenizer.decode(chunk)
        chunks.append(chunk_text)
    
    return chunks

# Connect to MongoDB Atlas
client = MongoClient(os.environ["MONGODB_API_KEY"])
db = client["Vector-store"]
collection = db["store-3"]

# Add embeddings to each document using the Hugging Face model, with chunking
for doc in documents:
    chunks = chunk_text(doc["content"])
    chunk_embeddings = [embed_text(chunk) for chunk in chunks]
    
    # Store each chunk with its embedding as a separate document
    doc_chunks = []
    for i, embedding in enumerate(chunk_embeddings):
        chunk_doc = {
            "filename": doc["filename"],
            "chunk_id": i,
            "content": chunks[i],
            "embedding": embedding,
        }
        doc_chunks.append(chunk_doc)
    
    if doc_chunks:
        collection.insert_many(doc_chunks)

print("Documents inserted successfully")

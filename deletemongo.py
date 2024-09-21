import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# MongoDB connection details from environment variables
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "GCBOT"
COLLECTION_NAME = "GCBOT"

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Delete all documents in the collection
delete_result = collection.delete_many({})

# Print the number of deleted documents
print(f"Deleted {delete_result.deleted_count} documents from the collection.")

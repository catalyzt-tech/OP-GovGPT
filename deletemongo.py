import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()
# MongoDB connection details from environment variables
MONGO_URI = os.getenv("MONGO_URI")
if MONGO_URI is None:
    raise ValueError("MONGO_URI environment variable is not set.")
DB_NAME = os.getenv("DB_NAME", "GCBOT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "GCBOT")
# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Delete all documents in the collection
try:
    delete_result = collection.delete_many({})
except Exception as e:
    print(f"Error occurred while deleting documents: {str(e)}")
    raise

# Print the number of deleted documents
print(f"Deleted {delete_result.deleted_count} documents from the collection.")

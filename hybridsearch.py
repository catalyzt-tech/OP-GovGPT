import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_cohere import CohereEmbeddings
import re

# Load environment variables from .env file
load_dotenv(override=True)


# Function to extract filenames from MongoDB results
def extract_filenames(document_str):
    pattern = r"'source': '([^']+)'"
    matches = re.findall(pattern, document_str)
    return list(set(matches))  # Remove duplicates more efficiently


# Function to process the filenames into URL format
def process_llm_response(llm_response):
    true_temp = set()  # Use a set for better performance
    for filename in llm_response:
        file_name = os.path.splitext(os.path.basename(filename))[0]
        url = file_name.replace("_", "/").replace("+", ":").replace(".txt", "")
        true_temp.add(url)
    return list(true_temp)  # Convert set back to list


# Function to perform weighted reciprocal rank fusion
def weighted_reciprocal_rank(doc_lists, weights=None):
    c = 60  # Constant from RRF paper

    if weights is None:
        weights = [1] * len(doc_lists)

    if len(doc_lists) != len(weights):
        raise ValueError("Number of rank lists must match the number of weights.")

    # Get all unique document IDs from the lists
    all_documents = set()
    for doc_list in doc_lists:
        for doc in doc_list:
            all_documents.add(doc["_id"])

    # Initialize RRF score for each document
    rrf_scores = {doc: 0.0 for doc in all_documents}

    # Calculate RRF scores
    for doc_list, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(doc_list, start=1):
            rrf_score = weight * (1 / (rank + c))
            rrf_scores[doc["_id"]] += rrf_score

    # Sort documents by RRF score in descending order
    sorted_docs = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    # Map sorted document IDs back to the original documents
    doc_id_map = {doc["_id"]: doc for doc_list in doc_lists for doc in doc_list}
    final_docs = [doc_id_map[doc_id] for doc_id in sorted_docs]

    return final_docs


# MongoDB connection function
def mongo_connect(uri):
    from pymongo.server_api import ServerApi

    if not uri:
        raise ValueError(
            "MongoDB URI is missing. Please set MONGO_URI in your environment variables."
        )

    try:
        client = MongoClient(uri, server_api=ServerApi("1"))
        client.admin.command("ping")
        print("Successfully connected to MongoDB!")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise e

    return client


# Function to generate embeddings for a query
def generate_embedding(text):
    api_key = os.environ.get("COHERE_API_KEY")

    if not api_key:
        raise ValueError(
            "Cohere API Key is missing. Please set COHERE_API_KEY in your environment variables."
        )

    model = CohereEmbeddings(
        model="embed-english-light-v3.0",
        cohere_api_key=api_key,
    )
    embedding = model.embed_query(text)
    return embedding


def atlas_hybrid_search(
    query, top_k, db_name, collection_name, vector_index_name, keyword_index_name
):
    # Generate embedding for the query
    query_vector = generate_embedding(query)

    # Vector search
    vector_results = mycollection.aggregate(
        [
            {
                "$vectorSearch": {
                    "queryVector": query_vector,
                    "path": "embedding",
                    "numCandidates": 25,
                    "limit": top_k,
                    "index": vector_index_name,
                },
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
    )
    vector_results_list = list(vector_results)

    # Keyword search
    keyword_results = mycollection.aggregate(
        [
            {
                "$search": {
                    "index": keyword_index_name,
                    "text": {"query": query, "path": "text"},
                }
            },
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            {"$limit": top_k},
        ]
    )
    keyword_results_list = list(keyword_results)

    # Collect sources only for citation
    extracted_filenames = extract_filenames(str(keyword_results_list))
    citation_data = process_llm_response(extracted_filenames)

    # Format document lists for RRF, excluding 'source'
    doc_lists = [vector_results_list, keyword_results_list]
    for i in range(len(doc_lists)):
        doc_lists[i] = [
            {"_id": str(doc["_id"]), "text": doc["text"], "score": doc["score"]}
            for doc in doc_lists[i]
        ]

    # Set weights: more importance to vector search (e.g., 2x weight for vector results)
    weights = [1, 1]

    # Apply rank fusion with weights
    fused_results = weighted_reciprocal_rank(doc_lists, weights)

    return [fused_results, citation_data]


# Load MongoDB URI from environment variables
uri = os.environ["MONGO_URI"]
client = mongo_connect(uri)

# Define MongoDB details
db_name = "GCBOT"
collection_name = "GCBOT"
vector_index_name = "GCBOT"
keyword_index_name = "keyword_search"

db = client.get_database(db_name)
mycollection = db.get_collection(collection_name)


# Function to perform hybrid research
def hybrid_research(query, top_k):
    # Check if the query is a dictionary and extract the string
    if isinstance(query, dict) and "question" in query:
        query_string = query["question"]
    elif isinstance(query, str):
        query_string = query
    else:
        raise ValueError(
            "Query must be a string or a dictionary containing a 'question' key."
        )

    # Debugging output
    print(
        f"Hybrid search called with query: {query_string}, type: {type(query_string)}"
    )

    result = atlas_hybrid_search(
        query_string,
        top_k,
        db_name,
        collection_name,
        vector_index_name,
        keyword_index_name,
    )

    return result


# print(hybrid_research("What is the capital of France?", 10))

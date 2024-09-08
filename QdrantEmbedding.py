import os
from dotenv import load_dotenv
import qdrant_client
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_chunks(text, filename):
    text_splitter = RecursiveCharacterTextSplitter(
        is_separator_regex=False,
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return [{"text": chunk, "filename": filename} for chunk in chunks]


def main():
    # Load environment variables
    load_dotenv()

    # Initialize Qdrant client
    client = qdrant_client.QdrantClient(
        api_key=os.getenv("QDRANT_API_KEY"),
        location=os.getenv("QDRANT_URL_KEY"),
    )

    client.set_model("sentence-transformers/all-MiniLM-L6-v2")
    # comment this line to use dense vectors only
    client.set_sparse_model("prithivida/Splade_PP_en_v1")

    # Create collection if it doesn't exist
    if not client.collection_exists("startupschunk2"):
        client.create_collection(
            collection_name="startupschunk2",
            vectors_config=client.get_fastembed_vector_params(),
            # comment this line to use dense vectors only
            sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
        )

    # Define the path where your text files are stored
    text_files_path = "AllData"

    metadata = []
    documents = []

    # Iterate through all text files and read content
    for file_name in os.listdir(text_files_path):
        file_path = os.path.join(text_files_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
                # Chunk the document
                chunks = get_chunks(content, file_name)

                # Collect chunks and corresponding metadata
                for chunk in chunks:
                    documents.append(chunk["text"])
                    metadata.append({"file_name": chunk["filename"]})

    # Add the chunks to the Qdrant collection
    client.add(
        collection_name="startupschunk2",
        documents=documents,
        metadata=metadata,
        parallel=3,  # Use all available CPU cores to encode data.
    )


if __name__ == "__main__":
    main()

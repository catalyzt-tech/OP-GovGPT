import shutil
import os

persist_directory = 'db'
# Delete the existing Chroma database
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

# Now reinitialize the Chroma vector store
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embeddings)

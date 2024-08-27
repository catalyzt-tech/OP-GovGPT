from crewai import Agent, Crew, Process, Task
from crewai_tools import DirectorySearchTool
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv()
# Set API keys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GROQ_API_KEY"] = os.getenv("groq_api")
os.environ["HF_TOKEN"] = os.getenv("HF_API_KEY")


persist_directory = "dbAlldata"



from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Initialize the LLM with the correct API key
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ["groq_api"],
    model_name="llama-3.1-8b-instant",
    temperature=0,
)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"}
)
# Initialize the search tool with directory search
search_tool = DirectorySearchTool(
    directory="Data2",
    config=dict(
        llm=dict(
            provider="groq",
            config=dict(
                model="llama-3.1-8b-instant",
            ),
        ),
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="sentence-transformers/all-MiniLM-L6-v2",
            ),
        ),
    ),
)

from pymongo import MongoClient

client = MongoClient(os.environ["MongoDB"])
db = client["Vector-store"]
collection = db["store-1"]

""" Retriever for citations """
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    text_key="content",  # The field where the text is stored
    embedding_key="embedding",  # The field where the embeddings are stored
    filename_key="filename",  # The field where the filename is stored
)

# Combine Tools - Define tool use for search and summarization
tools = [
    Tool(
        name="Search",
        func=lambda query: search_tool.run(query),  # Ensure the tool is callable
        description="Useful for searching documents directory and answering questions.",
    ),
]

prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    Question: {question}
    
    If you don't know the answer based on the information provided or retrieved, just say "I don't know."
    Don't try to make up an answer.
    """,
)


# Define the Research Agent with memory enabled and tools integrated
# Research and Verification Agent
Research_Agent = Agent(
    role="Research and Verification Agent",
    goal="Search through the directory to find relevant, accurate answers.",
    backstory=(
        "You are an assistant for question-answering tasks. "
        "Use the information present in the retrieved context to answer the question. "
        "Provide a clear, concise, and factually accurate answer. "
        "Verify the information and avoid any hallucinations or unsupported claims. "
        "If you don't know the answer, say 'I don't know'."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=tools,
    prompt=prompt_template,
    max_iter=3,
)

# Content Refinement Agent
Content_Agent = Agent(
    role="Content Refinement Agent",
    goal="Refine and polish the research output into a well-structured response.",
    backstory=(
        "You are a skilled editor and writer. Your task is to take the research output "
        "and refine it into a clear, engaging, and well-structured response. "
        "Ensure the key points are highlighted and the information flows logically."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
    max_iter=3,
)

# Research task
research_task = Task(
    description=(
        "Based on the user's question, extract information for the question {question} "
        "with the help of the tools. Use the Search tool to retrieve information from the directory. "
        "Verify the accuracy of the information and provide a clear, concise answer."
    ),
    expected_output=(
        "A clear, concise, and factually accurate answer to the user's question, "
        "based on the information retrieved from the directory. "
        "If you don't have the answer, return 'I don't know'."
    ),
    agent=Research_Agent,
)

# Content refinement task
refinement_task = Task(
    description=(
        "Take the research output and refine it into a well-structured, engaging response. "
        "Ensure the key points are highlighted and the information flows logically. "
        "Provide a brief conclusion or summary if appropriate."
    ),
    expected_output=(
        "A polished, well-structured response that clearly answers the user's question. "
        "The response should be engaging, easy to read, and include a brief conclusion if appropriate."
    ),
    context=[research_task],
    agent=Content_Agent,
)

# Updated Crew
rag_crew = Crew(
    agents=[Research_Agent, Content_Agent],
    tasks=[research_task, refinement_task],
    verbose=True,
    #process=Process.sequential,
    memory=True,
        embedder=dict(
        provider="huggingface",
        config=dict(
            model="sentence-transformers/all-MiniLM-L6-v2",
        ),
    ),
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,  # Ensure this is set to True
)


def process_llm_response(llm_response):
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


def handle_query(query):
    inputs = {"question": query}

    # Call the CrewAI process with the input question
    result = rag_crew.kickoff(inputs=inputs)

    if not result or "I don't know" in result:
        return "I don't know"

    print(result)

    # Invoke the LLM for additional query handling
    llm_response = qa_chain.invoke(query)
    process_llm_response(llm_response)


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


# Define a Pydantic model for the request body
class QuestionRequest(BaseModel):
    question: str


def serialize_crew_output(crew_output):

    return {
        "output": str(crew_output),
    }

import time

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    start_time = time.time()
    try:
        question = request.question
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")

        inputs = {"question": question}
        result = rag_crew.kickoff(inputs=inputs)

        serialized_result = serialize_crew_output(result)

        llm_response = qa_chain.invoke(question)
        link = process_llm_response(llm_response)
        print(f"Processing time: {time.time() - start_time} seconds")
        return {"result": serialized_result, "link": link}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run(app, host="0.0.0.0", port=5001)
    except KeyboardInterrupt:
        print("Server shut down gracefully")
    except Exception as e:
        print(f"An error occurred: {e}")

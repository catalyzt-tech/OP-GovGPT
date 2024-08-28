from crewai import Agent
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
import os
from crewai_tools import DirectorySearchTool

os.environ["GROQ_API_KEY"] = os.getenv("groq_api")

# Initialize the search tool with the specified directory and model configuration
search_tool = DirectorySearchTool(
    directory="docs-data-test",
    config=dict(
        llm=dict(
            provider="groq",
            config=dict(
                model="llama-3.1-70b-versatile",
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


class ResearchCrewAgents:

    def __init__(self):
        # Initialize the LLM to be used by the agents
        self.llm = ChatOpenAI(
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=os.environ["groq_api"],
            model_name="llama-3.1-70b-versatile",
            temperature=0,
            max_tokens=512,
        )
        # SELECT YOUR MODEL HERE
        self.selected_llm = self.llm

    def researcher(self):
        # Setup the tool for the Researcher agent
        tools = [
            Tool(
                name="Search",
                func=lambda inputs: search_tool.run(
                    inputs
                ),  # Use a lambda function to make `func` callable
                description="Useful for searching documents directory and answering questions.",
            ),
        ]

        return Agent(
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
            llm=self.selected_llm,
            tools=tools,  # Correctly pass the tools list
            max_iter=5,
        )

    def writer(self):
        # Setup the Writer agent
        return Agent(
            role="Content Writer",
            goal="Write engaging content based on the provided research or information.",
            backstory=(
                "You are a skilled writer who excels at turning raw data into captivating narratives."
                "Your task is to write clear, structured, and engaging content."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.selected_llm,
            max_iter=2,
        )

    def conclusion(self):
        # Setup the Conclusion agent
        return Agent(
            role="Conclusion Agent",
            goal="Generate a concise summary of the results from the previous tasks.",
            backstory=(
                "You are responsible for summarizing the key points and takeaways from the research and writing tasks. "
                "Your summary should be concise, informative, and capture the essence of the content."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.selected_llm,
            max_iter=2,
        )

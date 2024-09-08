from crewai import Agent
import os
from citation import Citation, HybridSearcher
from crewai_tools import BaseTool
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


class InfoSearchTool(BaseTool):
    name: str = "Info Search Tool"
    description: str = "Search data related information."

    def _run(self, query: str) -> str:
        # Retrieve relevant documents based on the query
        return HybridSearcher("startupschunk2").search(query)


# Initialize the search tool with the specified directory and model configuration
class ResearchCrewAgents:

    def __init__(self):
        # Initialize the LLM to be used by the agents
        self.cite = Citation()
        # SELECT YOUR MODEL HERE
        self.selected_llm = self.cite.llm

    def researcher(self):
        # Setup the tool for the Researcher agent
        tools = [InfoSearchTool()]
        return Agent(
            role="Research and Verification Agent",
            goal="Search through the data to find relevant and accurate answers.",
            backstory=(
                "You are an assistant for question-answering tasks."
                "Use the information present in the retrieved context to answer the question."
                "Provide a clear and concise answer."
                "Do not remove technical terms that are important for the answer, as this could make it out of context."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.selected_llm,
            tools=tools,  # Correctly pass the tools list
        )

    def writer(self):
        # Setup the Writer agent
        return Agent(
            role="Content Writer",
            goal="Write engaging content based on the provided research or information.",
            backstory=(
                "You are a professional in Optimism which is a Collective of companies, communities, and citizens working together to reward public goods and build a sustainable future for Ethereum."
                "Also, you are a skilled writer who excels at turning raw data into captivating narratives."
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
            goal="Generate a summary of the results from the previous tasks.",
            backstory=(
                "You are responsible for summarizing information from the research and writing tasks. "
                "Your summary should be concise, informative, and capture the essence of the content."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.selected_llm,
            max_iter=2,
        )

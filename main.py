from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware for FastAPI
import re
import os
from shared_state import SharedState
from crewai import Crew, Process
from agents import ResearchCrewAgents
from tasks import ResearchCrewTasks

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PORT"] = os.getenv("PORT", "10000")


class ResearchCrew:

    def __init__(self, inputs):
        self.inputs = inputs
        self.agents = ResearchCrewAgents()
        self.tasks = ResearchCrewTasks()

    def extract_filenames(self, document_str):
        pattern = r"'source': '([^']+)'"
        matches = re.findall(pattern, document_str)
        return list(set(matches))  # Remove duplicates more efficiently

    def process_llm_response(self, llm_response):
        true_temp = set()  # Use a set for better performance
        for filename in llm_response:
            file_name = os.path.splitext(os.path.basename(filename))[0]
            url = file_name.replace("_", "/").replace("+", ":").replace(".txt", "")
            true_temp.add(url)
        return list(true_temp)  # Convert set back to list

    def serialize_crew_output(self, crew_output):
        # If crew_output is already a string, no need to convert to string again
        return {"output": crew_output}

    async def run(self):
        researcher = self.agents.researcher()
        writer = self.agents.writer()

        research_task = self.tasks.research_task(researcher, self.inputs)
        writing_task = self.tasks.writing_task(writer, [research_task], self.inputs)

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=True,
        )
        result = await crew.kickoff_async(inputs=self.inputs)

        # self.rawsource = self.extract_filenames(SharedState().get_citation_data())
        # self.citation_data = self.process_llm_response(self.rawsource)

        self.serailized_result = self.serialize_crew_output(result)
        return {"result": self.serailized_result}
        # return {"result": self.serailized_result, "links": self.citation_data}

    async def run_discord(self):
        researcher = self.agents.researcher()
        writer = self.agents.writer()

        research_task = self.tasks.research_task(researcher, self.inputs)
        writing_task = self.tasks.writing_task_discord(
            writer, [research_task], self.inputs
        )

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=True,
        )
        result = await crew.kickoff_async(inputs=self.inputs)

        # self.rawsource = self.extract_filenames(SharedState().get_citation_data())
        # self.citation_data = self.process_llm_response(self.rawsource)

        self.serailized_result = self.serialize_crew_output(result)
        return {"result": self.serailized_result}
        # return {"result": self.serailized_result, "links": self.citation_data}


class QuestionRequest(BaseModel):
    question: str


USELESS_INFO_PHRASES = [
    "I don't know",
    "does not contain information",
    "does not contain any information",
    "any information",
    "Unfortunately",
    "Agent stopped due to iteration limit or time limit",
]


def has_useful_information(result):
    # Access the tasks_output from the result
    if "Agent stopped due to iteration limit or time limit" in result:
        return False
    return True


# def map_input(user_input):
#     text = user_input.lower()
#     if any(x in text for x in ["retrofunding 1", "retrofunding 2", "retrofunding 3"]):
#         return text.replace("retrofunding", "retropgf")

#     elif any(x in text for x in ["retropgf 4", "retropgf 5"]):
#         return text.replace("retropgf", "retrofunding")
#     else:
#         return user_input


app = FastAPI()

# Add CORS middleware for FastAPI

# Use environment variable to determine the environment (production or development)
is_production = os.getenv("ENV") == "production"
print("HERE IS is production ", is_production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        ["https://gptgov.app"]
        if is_production
        else ["http://localhost:3000", "http://127.0.0.1:3000"]
    ),
    allow_methods=["*"],
    allow_headers=["*"],
    # allow_credentials=True,
)


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question.strip()  # Strip any leading/trailing spaces
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        # mapped_question = map_input(question)
        # inputs = {"question": mapped_question}
        inputs = {"question": question}
        research_crew = ResearchCrew(inputs)
        result = await research_crew.run()
        crew_output = result["result"]["output"]  # This is a CrewOutput object

        # Check if it has useful information
        if has_useful_information(
            crew_output.raw
        ):  # Access 'raw' attribute of CrewOutput
            return result
        else:
            return {
                "result": "I cannot find any relevant information on this topic",
                "links": [],
            }
    except Exception as e:
        print(f"Error occurred: {e}")  # Log error for debugging
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/discord")
async def ask_question_discord(request: QuestionRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")

        inputs = {"question": question}
        research_crew = ResearchCrew(inputs)
        result = await research_crew.run_discord()
        # Access the raw output from the CrewOutput object
        crew_output = result["result"]["output"]  # This is a CrewOutput object

        # Check if it has useful information
        if has_useful_information(
            crew_output.raw
        ):  # Access 'raw' attribute of CrewOutput
            return result
        else:
            return {
                "result": "I cannot find any relevant information on this topic",
                "links": [],
            }
    except Exception as e:
        print(f"Error occurred: {e}")  # Log error for debugging
        raise HTTPException(status_code=500, detail="Internal Server Error")


import uvicorn

if __name__ == "__main__":
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 5001)),  # Default to 5001 if not set
            workers=int(
                os.environ.get("UVICORN_WORKERS", 4)
            ),  # Adjust based on testing
            log_level="info",  # Use 'debug' for more detailed logs if needed
            timeout_keep_alive=120,  # Increase timeout if necessary
        )
    except KeyboardInterrupt:
        print("Server shut down gracefully")
    except Exception as e:
        print(f"An error occurred: {e}")

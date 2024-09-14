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
        return matches

    def process_llm_response(self, llm_response):
        true_temp = []
        for filename in llm_response:
            file_name = os.path.splitext(os.path.basename(filename))[0]
            url = file_name.replace("_", "/").replace("+", ":").replace(".txt", "")
            if url not in true_temp:
                true_temp.append(url)
        return true_temp

    def serialize_crew_output(self, crew_output):
        return {"output": str(crew_output)}

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

        self.rawsource = self.extract_filenames(SharedState().get_citation_data())
        self.citation_data = self.process_llm_response(self.rawsource)

        self.serailized_result = self.serialize_crew_output(result)
        return {"result": self.serailized_result, "links": self.citation_data}

    async def run_discord(self):
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

        self.rawsource = self.extract_filenames(SharedState().get_citation_data())
        self.citation_data = self.process_llm_response(self.rawsource)

        self.serailized_result = self.serialize_crew_output(result)
        return {"result": self.serailized_result, "links": self.citation_data}


class QuestionRequest(BaseModel):
    question: str


USELESS_INFO_PHRASES = [
    "I don't know",
    "does not contain information",
    "does not contain any information",
    "any information",
    "Unfortunately",
]


def has_useful_information(output):
    return not any(phrase in output for phrase in USELESS_INFO_PHRASES)


def map_input(user_input):
    text = user_input.lower()
    if any(x in text for x in ["retrofunding 1", "retrofunding 2", "retrofunding 3"]):
        return text.replace("retrofunding", "retropgf")

    elif any(x in text for x in ["retropgf 4", "retropgf 5"]):
        return text.replace("retropgf", "retrofunding")
    else:
        return user_input


app = FastAPI()

# Add CORS middleware for FastAPI

# Use environment variable to determine the environment (production or development)
is_production = os.getenv("ENV") == "production"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gptgov.app/"] if is_production else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"] if is_production else ["*"],
    allow_headers=["Authorization", "Content-Type"] if is_production else ["*"],
)


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        mapped_question = map_input(question)
        inputs = {"question": mapped_question}
        research_crew = ResearchCrew(inputs)
        result = await research_crew.run()

        if has_useful_information(result["result"]):
            return result
        else:
            # If the result is not useful, don't return any links
            return {
                "result": "I cannot find any relevant information on this topic",
                "links": [],
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/discord")
async def ask_question_discord(request: QuestionRequest):
    try:
        question = request.question
        mapped_question = map_input(question)
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")

        inputs = {"question": mapped_question}
        research_crew = ResearchCrew(inputs)
        result = await research_crew.run_discord()
        if has_useful_information(result["result"]):
            return result
        else:
            # If the result is not useful, don't return any links
            return {
                "result": "I cannot find any relevant information on this topic",
                "links": [],
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import uvicorn

if __name__ == "__main__":
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 5001)),  # Default to 5001 if not set
            workers=int(os.environ.get("UVICORN_WORKERS", 4)),  # Default to 4 workers
        )
    except KeyboardInterrupt:
        print("Server shut down gracefully")
    except Exception as e:
        print(f"An error occurred: {e}")

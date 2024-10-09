from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from crewai import Crew, Process
from agents import ResearchCrewAgents
from tasks import ResearchCrewTasks
import logging


# Setup environment variables
def setup_environment():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PORT"] = os.getenv("PORT", "10000")


setup_environment()

logging.basicConfig(level=logging.INFO)


class ResearchCrew:
    def __init__(self, inputs):
        self.inputs = inputs
        self.agents = ResearchCrewAgents()
        self.tasks = ResearchCrewTasks()

    def serialize_crew_output(self, crew_output):
        """Serialize crew output to a dictionary with a 'raw' key."""
        if isinstance(crew_output, str):
            return {"raw": crew_output}
        elif isinstance(crew_output, dict):
            return {"raw": crew_output.get("raw", "")}
        return {"raw": getattr(crew_output, "raw", "")}

    async def run(self, is_discord=False):
        from hybridsearch import hybrid_research

        researcher = self.agents.researcher()
        writer = self.agents.writer()

        research_task = self.tasks.research_task(researcher, self.inputs)
        writing_task = (
            self.tasks.writing_task_discord if is_discord else self.tasks.writing_task
        )(writer, [research_task], self.inputs)

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=True,
        )

        try:
            # Kick off the crew task
            self.result = await crew.kickoff_async(inputs=self.inputs)
            logging.info(f"Result received from crew: {self.result}")
        except Exception as e:
            logging.error(f"Error during crew kickoff: {e}")
            raise HTTPException(status_code=500, detail="Error while running crew task")

        try:
            # Perform hybrid search
            self.citation = hybrid_research(self.inputs, 10)[1]
        except Exception as e:
            logging.error(f"Error during hybrid search: {e}")
            raise HTTPException(status_code=500, detail="Error during hybrid search")

        try:
            # Serialize the result
            self.serialized_result = self.serialize_crew_output(self.result)
            logging.info(f"Serialized result: {self.serialized_result}")
        except Exception as e:
            logging.error(f"Error serializing result: {e}")
            raise HTTPException(
                status_code=500, detail="Error while serializing crew output"
            )

        return {
            "result": self.serialized_result.get("raw", ""),
            "links": self.citation,
        }


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
    """Check if the result contains useful information."""
    return not any(phrase in result for phrase in USELESS_INFO_PHRASES)


app = FastAPI()

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        ["https://gptgov.app"]
        if os.getenv("ENV") == "production"
        else ["http://localhost:3000", "http://127.0.0.1:3000"]
    ),
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper function to process the question
async def process_question(question: str, is_discord: bool = False):
    research_crew = ResearchCrew({"question": question})
    result = await research_crew.run(is_discord)

    # Check if result["result"] is useful
    crew_output_raw = result.get("result", "")
    if not has_useful_information(crew_output_raw):
        return {
            "result": "I cannot find any relevant information on this topic",
            "links": [],
        }

    return result


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")

        return await process_question(question)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/discord")
async def ask_question_discord(request: QuestionRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")

        return await process_question(question, is_discord=True)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


import uvicorn

if __name__ == "__main__":
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 5001)),
            workers=int(os.environ.get("UVICORN_WORKERS", 4)),
            log_level="info",
            timeout_keep_alive=120,
        )
    except KeyboardInterrupt:
        logging.info("Server shut down gracefully")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

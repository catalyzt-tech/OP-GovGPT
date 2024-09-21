from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from crewai import Crew, Process
from agents import ResearchCrewAgents
from tasks import ResearchCrewTasks


# Setup environment variables
def setup_environment():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PORT"] = os.getenv("PORT", "10000")


setup_environment()


class ResearchCrew:
    def __init__(self, inputs):
        self.inputs = inputs
        self.agents = ResearchCrewAgents()
        self.tasks = ResearchCrewTasks()

    def serialize_crew_output(self, crew_output):
        return {"output": crew_output}

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
        self.result = await crew.kickoff_async(inputs=self.inputs)
        self.citation = hybrid_research(self.inputs, 5)[1]

        self.serialized_result = self.serialize_crew_output(self.result)
        return {"result": self.serialized_result, "links": self.citation}


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
    return "Agent stopped due to iteration limit or time limit" not in result


app = FastAPI()

# Add CORS middleware for FastAPI
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


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")

        research_crew = ResearchCrew({"question": question})
        result = await research_crew.run()
        crew_output = result["result"]["output"]

        if has_useful_information(crew_output.raw):
            return {"result": result}
        else:
            return {
                "result": "I cannot find any relevant information on this topic",
                "links": [],
            }
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/discord")
async def ask_question_discord(request: QuestionRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")

        research_crew = ResearchCrew({"question": question})
        result = await research_crew.run(is_discord=True)
        crew_output = result["result"]["output"]

        if has_useful_information(crew_output.raw):
            return {"result": result}
        else:
            return {
                "result": "I cannot find any relevant information on this topic",
                "links": [],
            }
    except Exception as e:
        print(f"Error occurred: {e}")
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
        print("Server shut down gracefully")
    except Exception as e:
        print(f"An error occurred: {e}")

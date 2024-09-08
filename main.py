import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import uvicorn
import re

# Import the necessary classes
from crewai import Crew, Process
from agents import ResearchCrewAgents
from tasks import ResearchCrewTasks
from citation import Citation  # Import the Citation class

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Default port on render is 10000
os.environ["PORT"] = os.getenv("PORT", "10000")

import io
from contextlib import redirect_stdout


class ResearchCrew:
    def __init__(self, inputs):
        self.inputs = inputs
        self.agents = ResearchCrewAgents()
        self.tasks = ResearchCrewTasks()

    def extract_filenames(self, document_str):
        # Define the regex pattern to extract all filenames
        pattern = r"'file_name': '([^']+)'"

        # Find all matches for the pattern in the document string
        matches = re.findall(pattern, document_str)

        # Return the list of extracted filenames
        return matches

    def serialize_crew_output(self, crew_output):
        return {"output": str(crew_output)}

    def run(self):
        researcher = self.agents.researcher()
        writer = self.agents.writer()
        conclude = self.agents.conclusion()

        research_task = self.tasks.research_task(researcher, self.inputs)
        writing_task = self.tasks.writing_task(writer, [research_task], self.inputs)
        conclude_task = self.tasks.conclusion_task(
            conclude, [writing_task], self.inputs
        )

        crew = Crew(
            agents=[researcher, writer, conclude],
            tasks=[research_task, writing_task, conclude_task],
            process=Process.sequential,
            verbose=True,
        )

        # result = crew.kickoff(inputs=self.inputs)
        # self.serailized_result = self.serialize_crew_output(result)
        # return {"result": self.serailized_result}
        # Capture logs
        log_capture = io.StringIO()
        with redirect_stdout(log_capture):
            result = crew.kickoff(inputs=self.inputs)

        logs = log_capture.getvalue()
        self.filenames = self.extract_filenames(logs)
        print(f"Extracted Filenames: {self.filenames}")
        self.serailized_result = self.serialize_crew_output(result)
        self.citation = Citation().process_llm_response(self.filenames)
        return {"result": self.serailized_result, "links": self.citation}

    def run_discord(self):
        researcher = self.agents.researcher()
        writer = self.agents.writer()
        conclude = self.agents.conclusion()

        research_task = self.tasks.research_task(researcher, self.inputs)
        writing_task = self.tasks.writing_task(writer, [research_task], self.inputs)
        conclude_task = self.tasks.discord_conclusion_task(
            conclude, [writing_task], self.inputs
        )

        crew = Crew(
            agents=[researcher, writer, conclude],
            tasks=[research_task, writing_task, conclude_task],
            process=Process.sequential,
            verbose=True,
        )

        # Capture logs
        log_capture = io.StringIO()
        with redirect_stdout(log_capture):
            result = crew.kickoff(inputs=self.inputs)

        logs = log_capture.getvalue()
        self.filenames = self.extract_filenames(logs)
        # print(f"Extracted Filenames: {filenames}")
        self.serailized_result = self.serialize_crew_output(result)
        self.citation = Citation().process_llm_response(self.filenames)
        return {"result": self.serailized_result, "links": self.citation}


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


app = FastAPI()


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    start_time = time.time()
    try:
        question = request.question
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")

        inputs = {"question": question}
        research_crew = ResearchCrew(inputs)
        result = research_crew.run()
        if has_useful_information(result["result"]):
            return result

        print(f"Processing time for CrewAI: {time.time() - start_time} seconds")
        return {
            "result": "I cannot find any relevant information on this topic",
            "links": [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/discord")
async def ask_question_discord(request: QuestionRequest):
    start_time = time.time()
    try:
        question = request.question
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")

        inputs = {"question": question}
        research_crew = ResearchCrew(inputs)
        result = research_crew.run()
        if has_useful_information(result["result"]):
            return result

        print(f"Processing time for CrewAI: {time.time() - start_time} seconds")
        return {
            "result": "I cannot find any relevant information on this topic",
            "links": [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PORT"]))
    except KeyboardInterrupt:
        print("Server shut down gracefully")
    except Exception as e:
        print(f"An error occurred: {e}")

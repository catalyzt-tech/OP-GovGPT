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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Default port on render is 10000
os.environ["PORT"] = os.getenv("PORT", "10000")


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

        research_task = self.tasks.research_task(researcher, self.inputs)
        writing_task = self.tasks.writing_task(writer, [research_task], self.inputs)

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff(inputs=self.inputs)
        self.serailized_result = self.serialize_crew_output(result)
        return {"result": self.serailized_result}

    def run_discord(self):
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

        result = crew.kickoff(inputs=self.inputs)
        self.serailized_result = self.serialize_crew_output(result)
        return {"result": self.serailized_result}


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

    elif any(x in text for x in ["retrofund"]):
        return text.replace("retrofund", "retrofunding")

    else:
        return user_input


app = FastAPI()


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    start_time = time.time()
    try:
        question = request.question
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        mapped_question = map_input(question)
        inputs = {"question": mapped_question}
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
        mapped_question = map_input(question)
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")

        inputs = {"question": mapped_question}
        mappedinput = map_input(inputs)
        research_crew = ResearchCrew(mappedinput)
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
        uvicorn.run(
            "main:app",  # Replace with the actual module name (e.g., 'main' if your file is main.py)
            host="0.0.0.0",
            port=int(os.environ["PORT"]),
            workers=4,  # Adjust based on your available CPU cores
        )
    except KeyboardInterrupt:
        print("Server shut down gracefully")
    except Exception as e:
        print(f"An error occurred: {e}")

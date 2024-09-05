import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import uvicorn

# Import the necessary classes
from crewai import Crew, Process
from agents import ResearchCrewAgents
from tasks import ResearchCrewTasks
from citation import Citation  # Import the Citation class

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Default port on render is 10000
os.environ["PORT"] = os.getenv("PORT", "10000")


class ResearchCrew:
    def __init__(self, inputs):
        self.inputs = inputs
        self.agents = ResearchCrewAgents()
        self.tasks = ResearchCrewTasks()

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

        return crew.kickoff(inputs=self.inputs)

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

        return crew.kickoff(inputs=self.inputs)


class QuestionRequest(BaseModel):
    question: str


def serialize_crew_output(crew_output):
    return {"output": str(crew_output)}

USELESS_INFO_PHRASES = [
    "I don't know",
    "does not contain information",
    "does not contain any information",
    "any information",
    "Unfortunately"
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

        serialized_result = serialize_crew_output(result)
        print(f"Processing time for CrewAI: {time.time() - start_time} seconds")
        links = []
        if has_useful_information(serialized_result['output']):
            citation = Citation()
            qa_chain = citation.qa_chain()
            llm_response = qa_chain(question)
            links = citation.process_llm_response(llm_response)
            print(f"Processing time for Link: {time.time() - start_time} seconds")
            return {"result": serialized_result, "links": links}
        return {"result": "I cannot find any relevant information on this topic", "links": links}

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
        result = research_crew.run_discord()

        serialized_result = serialize_crew_output(result)
        print(f"Processing time for CrewAI: {time.time() - start_time} seconds")

        links = []
        if has_useful_information(serialized_result['output']):
            citation = Citation()
            qa_chain = citation.qa_chain()
            llm_response = qa_chain(question)
            links = citation.process_llm_response(llm_response)
            print(f"Processing time for Link: {time.time() - start_time} seconds")
            return {"result": serialized_result, "links": links}
        return {"result": "I cannot find any relevant information on this topic", "links": links}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PORT"]))
    except KeyboardInterrupt:
        print("Server shut down gracefully")
    except Exception as e:
        print(f"An error occurred: {e}")

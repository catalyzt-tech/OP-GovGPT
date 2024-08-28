from crewai import Task


class ResearchCrewTasks:

    def research_task(self, agent, inputs):
        return Task(
            description=(
                f"Based on the {inputs['question']} question, extract information for the question {inputs['question']} "
                "with the help of the tools. Use the Search tool to retrieve information from the directory. "
                "Verify the accuracy of the information and provide a clear, concise answer."
            ),
            expected_output=(
                "A clear, concise, and factually accurate answer to the user's question, "
                "based on the information retrieved from the directory. "
                "If you don't have the answer, return 'I don't know'."
            ),
            agent=agent,
        )

    def writing_task(self, agent, context, inputs):
        return Task(
            description=(
                "Use the verified research information provided by the Research Agent."
                "Your task is to create a well-structured and engaging piece of content."
                "Focus on clarity, readability, and flow. The content should be suitable for"
                "the intended audience and the topic should be covered comprehensively."
                f"Use {context} and {inputs['question']} to guide the writing process."
            ),
            expected_output=(
                "A complete and engaging piece of content"
                "that is well-structured, easy to read, and aligns with the information provided."
                "The final content should be formatted and ready for publication."
            ),
            agent=agent,
            context=context,
        )

    def conclusion_task(self, agent, context, inputs):
        return Task(
            description=(
                "Generate a concise summary of the results from the writing_task tasks. "
                f"The conclusion should focus on the key points and provide a brief overview that related to {inputs['question']}. "
                "Keep the summary to one to six sentences."
            ),
            expected_output=(
                "A brief, one to six sentences summary that highlights the key takeaways from the previous tasks."
            ),
            agent=agent,
            context=context,
        )

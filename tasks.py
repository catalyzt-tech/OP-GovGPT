from crewai import Task


class ResearchCrewTasks:

    def research_task(self, agent, inputs):
        return Task(
            description=(
                f"Based on the {inputs['question']} question, extract information for the question {inputs['question']} "
                "with the help of the tools. Use the Search tool to retrieve information. "
                "Verify the accuracy of the information and provide a clear, concise answer."
            ),
            expected_output=(
                f"A clear, concise, and factually accurate answer to the user's question {inputs['question']},"
                "based on the information retrieved from the tool use. Don't make up an answer. If you don't know the answer, just say 'I don't know'."
                "Don't remove the key points and technical terms words such as OP Stack, onchain, superchain and so on, it can make the answer not related to the improvement of optimism ecosystem."
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
                "Ensure that the final content is formatted and ready for publication and use all the key points to write the answer."
            ),
            expected_output=(
                "A complete and engaging piece of content"
                "that is well-structured, easy to read, and aligns with the information provided."
                "The final content should be formatted and ready for publication. Don't make up an answer."
                f"The answer need to use the context {context} and write the best friendly answer related to the question {inputs['question']}"
                "Also, use the data that have the high like_count, trust_level be priority to write an answer, if it has these options in the retrived data. Also, don't forget to use the technical terms word to write."
            ),
            agent=agent,
            context=context,
        )

    def conclusion_task(self, agent, context, inputs):
        return Task(
            description=(
                f"Generate a concise summary of the results from the context {context} tasks. "
                f"The conclusion should focus on the key points and provide a brief overview that related to {inputs['question']}. "
            ),
            expected_output=(
                f"A brief summary that highlights the key of question {inputs['question']} from the previous tasks."
                f"The answer need to use the context {context} and write the best friendly answer conclusion related to the use all the key points to conclude, don't forget to use the technical terms word to conclude."
                "Use the content part in the data retrieved from the tool to write the answer."
            ),
            agent=agent,
            context=context,
        )

    def discord_conclusion_task(self, agent, context, inputs):
        return Task(
            description=(
                f"Generate a concise summary of the results from the context {context} tasks. "
                f"The conclusion should focus on the key points and provide a brief overview that related to {inputs['question']}. "
            ),
            expected_output=(
                f"A brief summary that highlights the key of question {inputs['question']} from the previous tasks."
                f"The answer need to use the context {context} and write the best friendly answer conclusion related to the question {inputs['question']} use all the key points to conclude"
                "Use the content part in the data retrieved from the tool to write the answer."
                "Make sure that the answer need to be concluded in maximum 8 sentences."
            ),
            agent=agent,
            context=context,
        )

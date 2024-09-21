from crewai import Task


class ResearchCrewTasks:

    def research_task(self, agent, inputs):
        return Task(
            description=(
                f"Based on the {inputs['question']} question, extract information for the question {inputs['question']} "
                "with the help of the tools. Use the Search tool to retrieve information."
                "Verify the accuracy of the information and provide a clear, concise answer."
            ),
            expected_output=(
                f"A clear, concise, and factually accurate answer to the user's question {inputs['question']},"
                "based on the information retrieved from the tool use. Don't make up an answer."
                f"If the {inputs['question']} question not related information retrieved from the tool just say 'The answer can be incorrect or not related to the question', but don't be too strict, if it not 100% related, 80% is acceptable."
                "Ensure to include key points and technical terms such as OP Stack, onchain, superchain, etc., to maintain relevance to the optimism ecosystem."
            ),
            agent=agent,
            async_execution=True,
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
                "A complete and engaging piece of content, write in 8 sentences."
                "that is well-structured, easy to read, and aligns with the information provided."
                "The final content should be formatted and ready for publication. Don't make up an answer."
                f"If the {inputs['question']} question not related information retrieved from the research agents just say 'Unfortunately, I could not find any relevant information on this topic'."
                f"The answer need to use the context {context} and write the best friendly answer related to the question {inputs['question']}"
                "Prioritize data with high like_count and trust_level when writing the answer, and ensure to incorporate technical terms."
            ),
            agent=agent,
            context=context,
            # async_execution=True,
        )

    def writing_task_discord(self, agent, context, inputs):
        return Task(
            description=(
                "Use the verified research information provided by the Research Agent."
                "Your task is to create a well-structured and engaging piece of content."
                "Focus on clarity, readability, and flow. The content should be suitable for"
                "the intended audience and the topic should be covered comprehensively."
                "Ensure that the final content is formatted and ready for publication and use all the key points to write the answer."
            ),
            expected_output=(
                "A complete and engaging piece of content, concluded in maximum 5 sentences."
                "that is well-structured, easy to read, and aligns with the information provided."
                "The final content should be formatted and ready for publication. Don't make up an answer."
                f"If the {inputs['question']} question not related information retrieved from the research agents just say 'Unfortunately, I could not find any relevant information on this topic'."
                f"The answer need to use the context {context} and write the best friendly answer related to the question {inputs['question']}"
                "Prioritize data with high like_count and trust_level when writing the answer, and ensure to incorporate technical terms."
            ),
            agent=agent,
            context=context,
            # async_execution=True,
        )

    def conclusion_task(self, agent, context, inputs):
        return Task(
            description=(
                f"Generate a concise summary of the results from the context {context} tasks. "
                f"The conclusion should focus on the key points and provide a brief overview that related to {inputs['question']}. "
            ),
            expected_output=(
                f"A brief summary that highlights the key of question {inputs['question']} from the previous tasks."
                f"Ensure the conclusion utilizes all key points and technical terms, providing a comprehensive summary that is contextually relevant to {inputs['question']}"
                "Use the content part in the data retrieved from the tool to write the answer."
            ),
            agent=agent,
            context=context,
            # async_execution=True,
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
                "Make sure that the answer need to be concluded in maximum 6 sentences."
            ),
            agent=agent,
            context=context,
            # async_execution=True,
        )

from openai import OpenAI
from answerbot.reactor import llm_react

from answerbot.subtasks.table_of_contents import ReadFromTableOfContents
from .toc import get_eu_act_toc


def answer(
        client: OpenAI,
        model: str,
        question: str,
        energy: int=30,
        ) -> list:
    table_of_contents = get_eu_act_toc()
    read_toc = ReadFromTableOfContents(
        client,
        model,
        question=question,
        document_toc=table_of_contents,
        docstring='Read from EU Artificial Intelligence Act'
        )

    toolbox = [
            read_toc.read_from_table_of_contents
            ]

    result = llm_react(
            client=client,
            model=model,
            energy=energy,
            domain='EU Artificial Intelligence Act',
            toolset=toolbox,
            question=question
            )

    return result
    

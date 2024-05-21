from llm_easy_tools.schema_generator import get_tool_defs
from openai import OpenAI

from .react import llm_react, cost, use_limit

from answerbot.subtasks.table_of_contents import ReadFromTableOfContents
from .toc import get_eu_act_toc


@use_limit(3)
@cost(10)
class ReadEUAct(ReadFromTableOfContents):
    ''' Read from EU Artificial Intelligence Act'''

    __name__ = 'read_eu_act'


def answer(
        client: OpenAI,
        model: str,
        question: str,
        energy: int=30,
        ) -> list:
    table_of_contents = get_eu_act_toc()

    toolbox = [
            ReadEUAct(
                client=client,
                model=model,
                question=question,
                document_toc=table_of_contents,
                read_article_cost=3,
                )
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
    

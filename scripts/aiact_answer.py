from collections.abc import Callable
import logging
from typing import Collection, Iterable, Any, TypeAlias

logging.basicConfig(level=logging.INFO)

from openai import OpenAI
from answerbot.aiact.answer import answer
from answerbot.aiact.react import Finish
from answerbot.subtasks.table_of_contents import ReadFromTableOfContentsResult
import os


ReflectionFormatter: TypeAlias = Callable[[list], str|Any]


def format_reactor_results(
        formatters: Collection[ReflectionFormatter],
        question: str, 
        reflections: list) -> str:
    chunks = [f'# Question:\n\n{question.strip()}'] + [formatter(reflections) for formatter in formatters]
    return '\n\n'.join(chunks)


def format_answer(reflections: list) -> str|None:
    for reflection in reflections:
        if not isinstance(reflection, Finish):
            continue
        return f'# Answer\n\n{reflection.answer.strip()}'


def format_read_from_table_of_contents(reflections: list) -> str|None:
    chunks = [
            f'## {article.article.title.strip()}\n\n {article.reflection.summary}'
            for reflection in reflections
            if isinstance(reflection, ReadFromTableOfContentsResult)
            for article in reflection.articles
            ]

    if not chunks:
        return None

    chunks_text = '\n\n'.join(chunks)

    return f'# Information gathered\n\n{chunks_text}'



question = '''
How is transparency defined in the AI Act and what transparency requirements apply to low-risk Ai systems?
'''


client = OpenAI(
  base_url="http://oai.hconeai.com/v1", 
  default_headers= { 
    "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}",
  },
)


result = answer(
        client,
        'gpt-4o',
        question=question,
        energy=100,
        )


formatted = format_reactor_results(
        [format_answer, format_read_from_table_of_contents],
        question,
        result
        )
print(formatted)

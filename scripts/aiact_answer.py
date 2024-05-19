import logging

logging.basicConfig(level=logging.INFO)

from openai import OpenAI
from answerbot.aiact.react import answer
from answerbot.reactor import Finish
from answerbot.subtasks.table_of_contents import ReadFromTableOfContentsResult
import os


def format_reactor_results(question: str, results: list) -> str:
    chunks = []
    chunks.append(f'# Question\n\n{question}')

    for item in results:
        if isinstance(item, Finish):
            chunks.append(f'# Answer\n\n{item.answer}')

    relevant_info = []

    for item in results:
        if isinstance(item, ReadFromTableOfContentsResult):
            for article in item.articles:
                relevant_info.append(
                        f'## {article.article.title}\n\n {article.reflection.summary}'
                        )

    relevant_info_text = '\n\n'.join(relevant_info)
    chunks.append(f'# Information gathered\n\n{relevant_info_text}')

    return '\n\n'.join(chunks)


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


formatted = format_reactor_results(question=question, results=result)
print(formatted)

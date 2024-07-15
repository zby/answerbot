import logging
import os
import httpx
from openai import OpenAI
from answerbot.qa_processor import QAProcessor
from answerbot.tools.aae import AAESearch
from pprint import pprint
import click
import json
import litellm

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logging.getLogger('LiteLLM').setLevel(logging.WARNING)

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

question = '''
How is transparency defined in the AI Act and what transparency requirements apply to low-risk Ai systems?
'''

question = "Does the deployment of an LLM acting as a proxy to optimize SQL queries fall within the regulatory scope of the EU’s AI Act?"


@click.command()
@click.option('--max-llm-calls', '-m', type=int, default=5)
def main(local: str|None=None, max_llm_calls: int=5):
    app = QAProcessor(
        model="claude-3-5-sonnet-20240620",
        toolbox=[AAESearch(chunk_size=400)],
        max_iterations=max_llm_calls,
        prompt_templates_dirs=[
            "answerbot/templates/common",
            "answerbot/templates/aiact",
        ],
    )

    print()
    print(app.process(question))

if __name__ == '__main__':
    main()

import httpx
from answerbot.aiact import run_reactor, format_results
from answerbot.replay_client import ReplayClient

from dotenv import load_dotenv
from openai import OpenAI
import logging


logging.basicConfig(level=logging.DEBUG)

load_dotenv()
client = OpenAI(
        timeout=httpx.Timeout(70, read=60.0, write=20.0, connect=6.0)
        )

client = ReplayClient('data/messages.json')

if __name__ == '__main__':
    question = '''
    How is transparency defined in the AI Act and what transparency requirements apply to low-risk Ai systems?
    '''

    result = run_reactor(model='gpt-4-turbo', client=client, question=question,energy=200)

    print(format_results(result))

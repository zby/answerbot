import httpx
from answerbot.aiact import AiActReactor, format_results
from dotenv import load_dotenv
from openai import OpenAI
import logging

logging.basicConfig(level=logging.DEBUG)

openai_client = OpenAI(
        timeout=httpx.Timeout(70, read=60.0, write=20.0, connect=6.0)
        )
load_dotenv()


if __name__ == '__main__':
    question = '''
    How is transparency defined in the AI Act and what transparency requirements apply to low-risk Ai systems?
    '''

    reactor = AiActReactor(model='gpt-4-turbo', client=openai_client, question=question,energy=200)
    result = reactor()

    print(format_results(result))

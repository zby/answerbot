import logging
import httpx
from openai import OpenAI
from answerbot.react import get_answer
from answerbot.tools.aaext import EUAIAct
from pprint import pprint

from dotenv import dotenv_values

logging.basicConfig(level=logging.INFO)


question = '''
How is transparency defined in the AI Act and what transparency requirements apply to low-risk Ai systems?
'''


def main():
    env = dotenv_values('.env')
    client = OpenAI(
         timeout=httpx.Timeout(70.0, read=60.0, write=20.0, connect=6.0),
         api_key=env['OPENAI_API_KEY'],
         base_url="https://oai.hconeai.com/v1",
         default_headers={
             "Helicone-Auth": f"Bearer {env['HELICONE_API_KEY']}",
         }
    )
    config = {
        "chunk_size": 400,
        "prompt_class": 'NERP',
        #"prompt_class": 'AAE',
        "max_llm_calls": 7,
        "model": "gpt-4o",
        "question_check": 'None',
        'reflection': 'ShortReflectionDetached',
        'tool': EUAIAct,
        #'tool': AAESearch,
    }
    reactor = get_answer(question, config, client)
    print(reactor.trace.generate_report())
    print()
    print(str(reactor.what_have_we_learned))
    print()
    pprint(reactor.soft_errors)
    with open('data/trace.py', 'w') as file:
        file.write(repr(reactor.trace))
#    print(format_markdown(reactor.conversation))


if __name__ == '__main__':
    main()

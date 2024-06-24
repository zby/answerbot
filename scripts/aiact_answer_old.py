import logging
import httpx
from openai import OpenAI
from answerbot.react import LLMReactor
from answerbot.tools.aae import AAESearch
from pprint import pprint

from dotenv import dotenv_values

logging.basicConfig(level=logging.INFO)


question = '''
How is transparency defined in the AI Act and what transparency requirements apply to low-risk Ai systems?
'''


def _sys_prompt(max_llm_calls):
    return f"""
    Please answer the following question about the European Artificial Intelligence Act.
    You are provided with the tools you need to search across the act, but remember, you
    only have {max_llm_calls-1} attempts to use them.

    When you've found a document, you can use lookup and lookup_next functions to search for
    keywords in it.
    """


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

    reactor = LLMReactor(
            model='gpt-3.5-turbo',
            toolbox=[AAESearch(chunk_size=400)],
            max_llm_calls=7,
            client=client,
            get_system_prompt=_sys_prompt,

            )
    trace = reactor.process(question)
    print(trace.generate_report())
    print()
    print(str(trace.what_have_we_learned))
    print()
    pprint(trace.soft_errors)


if __name__ == '__main__':
    main()

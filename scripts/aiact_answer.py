import logging
import os
import httpx
from openai import OpenAI
from answerbot.react import LLMReactor, get_answer
from answerbot.tools.aaext import DocumentSection, EUAIAct, get_eu_act_toc, get_eu_act_toc_raw, retrieve, sanitize_string
from pprint import pprint
import click
import json

from dotenv import dotenv_values

logging.basicConfig(level=logging.INFO)




question = '''
How is transparency defined in the AI Act and what transparency requirements apply to low-risk Ai systems?
'''


def _get_toc_local(folder: str) -> DocumentSection:
    def _retrieve(title) -> str:  # type: ignore
        sanitized_title =sanitize_string(title)
        path = os.path.join(folder, f'{sanitized_title}.md')
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    with open(os.path.join(folder, 'toc.json'), 'r', encoding='utf-8') as f:
        raw = json.load(f)

    return get_eu_act_toc(raw, lambda *args: _retrieve(args[0]))


def _get_toc_web() -> DocumentSection:
    return get_eu_act_toc(
            get_eu_act_toc_raw(),
            lambda *args: retrieve(args[2])
            )



@click.command()
@click.option('--local', '-l', type=str, default=None)
@click.option('--max-llm-calls', '-m', type=int, default=7)
def main(local: str|None=None, max_llm_calls: int=7):
    def _sys_prompt(max_llm_calls):
        return f"""
        Please answer the following question, using the tool available.
        You only have {max_llm_calls-1} to the tools.
        """


    toc = _get_toc_web() if not local else _get_toc_local(local)

    env = dotenv_values('.env')
    client = OpenAI(
         timeout=httpx.Timeout(70.0, read=60.0, write=20.0, connect=6.0),
         api_key=env['OPENAI_API_KEY'],
         base_url="https://oai.hconeai.com/v1",
         default_headers={
             "Helicone-Auth": f"Bearer {env['HELICONE_API_KEY']}",
         }
    )

    reactor = LLMReactor.create_reactor(
            model='gpt-4o',
            toolbox=[EUAIAct(toc)],
            max_llm_calls=max_llm_calls,
            client=client,
            question=question,
            sys_prompt=_sys_prompt,
            question_checks=[],
            )

    reactor.process()
    print(reactor.generate_report())
    print()
    print(str(reactor.what_have_we_learned))
    print()
    pprint(reactor.soft_errors)


if __name__ == '__main__':
    main()

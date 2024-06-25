import logging
import os
import httpx
from openai import OpenAI
from answerbot.react import LLMReactor 
from answerbot.tools.aaext import DocumentSection, EUAIAct, get_eu_act_toc, get_eu_act_toc_raw, retrieve, sanitize_string
from pprint import pprint
import click
import json

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)




question = '''
How is transparency defined in the AI Act and what transparency requirements apply to low-risk Ai systems?
'''

question = "Does the deployment of an LLM acting as a proxy to optimize SQL queries fall within the regulatory scope of the EU’s AI Act?"

model = 'claude-3-5-sonnet-20240620'
model = 'gpt-4o'

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
    def _sub_sys_prompt(max_llm_calls):
        return f"""
        Please answer the following question, using the tool available.
        You only have {max_llm_calls-1} to the tools.
        """


    toc = _get_toc_web() if not local else _get_toc_local(local)

    def _main_sys_prompt(max_llm_calls):
        return f"""
Please answer the following user question. You can get help from an assistant with access to the EU AI Act
- by calling 'delegate' function and passing the question you want to ask him.

You need to carefully divide the work into tasks that would require the least amount of access to the EU AI Act,
and then delegate them to the assistant.
The questions you ask the assistant need to be as simple and specific as possible and they should have
a short answer, don't ask the assistant to retrieve some long piece of information, instead ask him a question
that can be answered based on the information that can be retrieved. Occasionally you can ask him to summarize
some information for you, but then always tell him what is the goal of that summarisation.
You can call finish when you think you have enough information to answer the question.
You can delegate only {max_llm_calls - 1} tasks to the assistant.
Here is the general structure of the EU AI Act:

----
{toc.to_string()}
----

"""

    sub_reactor = LLMReactor(
        model=model,
        toolbox=[EUAIAct(toc)],
        max_llm_calls=max_llm_calls,
        get_system_prompt=_sub_sys_prompt,
    )


    def delegate_to_expert(question: str):
        """
        Delegate the question to a wikipedia expert.
        """

        print(f'Delegating question: "{question}" to wikipedia expert')

        trace = sub_reactor.process(question)
        return trace.answer

    reactor = LLMReactor(
        model=model,
        toolbox=[delegate_to_expert],
        max_llm_calls=7,
        get_system_prompt=_main_sys_prompt,
        question_checks=["Please analyze the user question and find the first step in answering it - a task to delegate to a researcher with access to the EU AI Act, that would require the least amount of information retrievals. Think step by step."],
    )

    trace = reactor.process(question)
    print(trace.generate_report())
    print()
    print(str(trace.what_have_we_learned))
    print()
    pprint(trace.soft_errors)


if __name__ == '__main__':
    main()
m
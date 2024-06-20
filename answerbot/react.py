import time
import logging
import copy
from typing import Literal, Union, List, Dict, Annotated, Optional, Callable, Any, Protocol, Iterable, runtime_checkable
from pprint import pprint
from dataclasses import dataclass
import traceback

from openai.types.chat.chat_completion import ChatCompletionMessage

from .prompt_templates import QUESTION_CHECKS, PROMPTS, REFLECTIONS 

from llm_easy_tools import process_response, get_tool_defs, ToolResult
from answerbot.tools.wiki_tool import Observation, InfoPiece
from answerbot.clean_reflection import ReflectionResult, KnowledgeBase
from answerbot.trace import Trace, Question

# Configure basic logging
logging.basicConfig(level=logging.ERROR)

# Get a logger for the current module
logger = logging.getLogger(__name__)

@dataclass
class Answer:
    answer: str
    answer_short: str
    reasoning: str

    def normalized_answer(self):
        answer = self.answer
        answer = answer.strip(' \n.\'"')
        answer = answer.replace('’', "'")  # Replace all curly apostrophes with straight single quotes
        answer = answer.replace('"', "'")  # Replace all double quotes with straight single quotes
        if answer.lower() == 'yes' or answer.lower() == 'no':
            answer = answer.lower()
        return answer

    def __str__(self):
        return f'{self.normalized_answer()}\n\nReasoning: {self.reasoning}'


@runtime_checkable
class HasLLMTools(Protocol):
    def get_llm_tools(self) -> Iterable[Callable]:
        pass

@dataclass
class SubReactorResult:
    report: str
    trace: Trace
    reflection_prompt: str

    def __str__(self):
        return self.report

class LLMReactor:
    class LLMReactorError(Exception):
        pass

    def __init__(self, 
                model: str,
                toolbox: list[Callable|HasLLMTools],
                trace: Trace,
                max_llm_calls: int,
                client: object,
                question_checks=None,
                ):
        self.model = model
        self.toolbox = toolbox
        self.trace = trace
        self.max_llm_calls = max_llm_calls
        self.client = client
        self.question_checks = [] if question_checks is None else question_checks

        self.step = 0
        self.answer = None
        self.soft_errors = []
        self.reflection_prompt = []
        self.what_have_we_learned = KnowledgeBase()

    def get_tools(self) -> list[Callable]:
        tools = [self.finish]  # Note for the AI assistant - always include the finish tool
        if self.step < self.max_llm_calls + 1:
            for item in self.toolbox:
                if isinstance(item, HasLLMTools):
                    new_tools = item.get_llm_tools()
                    tools.extend(new_tools)
                else:
                    tools.append(item)
        return tools


    def openai_query(self, messages, schemas=[]):
        args = {
            'model': self.model,
            'messages': messages
        }

        if len(schemas) > 0:
            args['tools'] = schemas
            if len(schemas) == 1:
                args['tool_choice'] = {'type': 'function', 'function': {'name': schemas[0]['function']['name']}}
            else:
                args['tool_choice'] = "auto"

        completion = self.client.chat.completions.create( **args )

        if len(schemas) > 0:
            if not completion.choices[0].message.tool_calls:
                stack_trace = traceback.format_stack()
                self.soft_errors.append(f"No function call:\n{stack_trace}")

        return completion

    def query_and_process(self):
        tools = self.get_tools()
        schemas = get_tool_defs(tools)
        response = self.openai_query(self.trace.to_messages(), schemas)
        message = response.choices[0].message
        self.trace.append(message)
        results = process_response(response, tools)
        for result in results:
            if result.error is not None:
                print(result.error)
                raise self.LLMReactorError(result.stack_trace)
            self.trace.append(result)
        for result in results:
            if isinstance(result.output, Observation) and result.output.reflection_needed():
                self.clean_context_reflection(result)
        return results

    def base_reflection(self, result):
        # In clean context reflection we cannot ask the llm to plan - because it does not get the information retrieved previously.
        # But we can contrast the reflection with previous data ourselves - and for example remove links that were already retrieved.


        sysprompt = f"""You are a researcher working on the following user question:
{self.trace.user_question()}
"""

        new_trace = Trace()

        new_trace.append({'role': 'system', 'content': sysprompt})
        new_trace.append({'role': 'user', 'content': result.output.reflection_prompt})
        response = self.openai_query(new_trace.to_messages(), [])
        reflection_string = "**My Notes**\n" + response.choices[0].message.content
        message = { 'role': 'user', 'content': reflection_string }
        new_trace.result = message
        self.trace.append(new_trace)


    def clean_context_reflection(self, result):
        # In clean context reflection we cannot ask the llm to plan - because it does not get the information retrieved previously.
        # But we can contrast the reflection with previous data ourselves - and for example remove links that were already retrieved.

        trace = Trace()

        learned_stuff = f"\n\nSo far we have some notes on the following urls:{self.what_have_we_learned.learned()}" if not self.what_have_we_learned.is_empty() else ""
        jump_next = f"\n- jump to the next occurrence of the looked up keyword" if result.name == 'lookup' or result.name == 'next' else ""
        sysprompt = f"""You are a researcher working on the following user question:
{self.trace.user_question()}{learned_stuff}

You need to review the information retrieval recorded below."""

#. You need to decide if the next action should be:
#- read_more information from the current position on the current page
#- lookup a new keyword on the current page{jump_next}
#- go to a new url 
#- make a new wikipedia search
#- finish with an answer to the user question
#To save potential sources of new information please add their urls to the new_sources list."""
        user_prompt = str(result.output)
        trace.append({'role': 'system', 'content': sysprompt})
        trace.append({'role': 'user', 'content': user_prompt})
        schemas = get_tool_defs([ReflectionResult])
        response = self.openai_query(trace.to_messages(), schemas)
        message = response.choices[0].message
        trace.append(message)
        results = process_response(response, [ReflectionResult])
        if len(results) > 1:
            self.soft_errors.append(f"More than one reflection result")
        new_result = results[0]
        trace.append(new_result)
        if new_result.error is not None:
            raise self.LLMReactorError(new_result.stack_trace)
        reflection = new_result.output
        knowledge_piece = reflection.extract_knowledge(result.output)
        self.what_have_we_learned.add_knowledge_piece(knowledge_piece)
        reflection.remove_checked_urls(self.what_have_we_learned.urls())
        reflection_string = f"current url: {knowledge_piece.url}\n"
        if len(reflection.new_sources) > 0 or not knowledge_piece.is_empty():
            reflection_string += f"{str(knowledge_piece)}\n"
            if len(reflection.new_sources) > 0:
                reflection_string += f"Discovered new sources: {reflection.new_sources}"
        print(reflection_string)
        self.trace.append(trace)

        new_trace = Trace()

        available_tools_indented = "\n".join([f"  - {tool}" for tool in result.output.available_tools])
        second_user_prompt = f"""Ny notes:
{reflection_string}

What would you do next?
Please analyze the retrieved data and check if you have enough information to answer the user question. Explain your reasoning.

If you still need more information to retrieve please check if it is probable that that needed information is on current page in a not yet retrieved fragment or if you need to get another page using search or get_url. Then think carefully what the next retrieval action should be out of the available options:
{available_tools_indented}
  - finish: Ends the retrieval and answers the question
"""
        new_trace.append({'role': 'system', 'content': sysprompt})
        new_trace.append({'role': 'user', 'content': user_prompt})
        new_trace.append({'role': 'user', 'content': second_user_prompt})
        response = self.openai_query(new_trace.to_messages(), [])
        second_reflection = response.choices[0].message.content
        reflection_string = "**My Notes**\n" + reflection_string + "\n\nHmm what I could do next?\n\n" + second_reflection
        message = { 'role': 'user', 'content': reflection_string }
        new_trace.result = message
        self.trace.append(new_trace)


    def process(self):
        self.analyze_question()
        while self.answer is None and self.step <= self.max_llm_calls + 1:
            self.query_and_process()
            self.step += 1
            if self.step == self.max_llm_calls:
                step_info = "\n\nThis was the last data you can get - in the next step you need to formulate your answer"
            else:
                step_info = f"\n\nThis was {self.step} out of {self.max_llm_calls} calls for data."
            self.trace.append({'role': 'system', 'content': step_info})

#            if 'gpt-4' in self.model:
#                time.sleep(20)


    def analyze_question(self):
        for query in self.question_checks:
            question_check = { 'role': 'user', 'content': query }
            logger.info(str(question_check))
            trace = self.trace
            trace.append(question_check)
            response = self.openai_query(trace.to_messages())
            message = response.choices[0].message
            trace.append(message)

    def finish(self,
               answer: Annotated[str, "The answer to the user's question"],
               answer_short: Annotated[str, "A short version of the answer"],
               reasoning: Annotated[str, "The reasoning behind the answer. Think step by step. Mention all assumptions you make."],
    ):
        """
        Finish the task and return the answer.
        """
        answer = Answer(answer, answer_short, reasoning)
        self.answer = answer
        return answer

    def generate_report(self) -> str:
        report = f'''
The answer to the question:"{self.trace.user_question()}" is:
{str(self.answer)}
'''
        return report

    @classmethod
    def create_reactor(self, 
                       sys_prompt: Callable[[int], str],
                       question: str,
                       toolbox: list[Callable|HasLLMTools],
                       max_llm_calls: int,
                       client: object,
                       model: str,
                       question_checks: list[str]
                       ):
        trace = Trace()
        trace.append({'role': 'system', 'content': sys_prompt(max_llm_calls)})
        trace.append(Question(question))
        reactor = LLMReactor(model, toolbox, trace, max_llm_calls, client, question_checks)
        return reactor


def get_answer(question, config, client: object):
    print("\n\n<<< Question:", question)
    tool_class = config['tool']
    tool = tool_class(chunk_size=config['chunk_size'])
    toolbox = [tool]
    sys_prompt = PROMPTS[config['prompt_class']]
    initial_trace = Trace()
    initial_trace.append({'role': 'system', 'content': sys_prompt(config['max_llm_calls'], '')})
    initial_trace.append(Question(question))
    question_checks = QUESTION_CHECKS[config['question_check']] 
    reactor = LLMReactor(
        config['model'], toolbox, initial_trace, config['max_llm_calls'], client,
        question_checks=question_checks
    )
    reactor.process()
    return reactor

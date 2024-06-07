import json
import time
import logging
import copy
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Literal, Union, List, Dict, Annotated, Optional, Callable, Any, Protocol, Iterable, runtime_checkable
from pprint import pprint
from dataclasses import dataclass

from openai.types.chat.chat_completion import ChatCompletionMessage

from .prompt_templates import QUESTION_CHECKS, PROMPTS, REFLECTIONS 

from llm_easy_tools import process_response, get_tool_defs, ToolResult
from answerbot.tools.wiki_tool import Observation, InfoPiece
from answerbot.clean_reflection import ReflectionResult, KnowledgeBase

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Get a logger for the current module
logger = logging.getLogger(__name__)

class Trace:
    def __init__(self, entries = None, user_question=None):
        self.entries = [] if entries is None else entries
        self.user_question = user_question

    def add_entry(self, entry):
        self.entries.append(entry)

    def add_system_message(self, content):
        self.entries.append({ 'role': 'system', 'content': content })

    def add_user_question(self, content):
        self.entries.append({ 'role': 'user', 'content': f"Question: {content}" })
        self.user_question = content

    def to_messages(self) -> List[Dict]:
        """
        Returns:
        List[Dict]: A list of dictionaries representing the messages and tool results.
        """
        all_messages = []
        for entry in self.entries:
            if isinstance(entry, dict):
                all_messages.append(entry)
            elif isinstance(entry, BaseModel):
                all_messages.append(entry.model_dump())
            elif isinstance(entry, ToolResult):
                all_messages.append(entry.to_message())
            else:
                raise ValueError(f"Invalid entry type: {type(entry)}")
        return all_messages

    def __repr__(self):
        return f"Trace(entries={self.entries}, user_question={self.user_question!r})"

    def generate_report(self):
        """
        Generates a report from a Trace object containing the user question, the answer, and the list of document quotes used.

        Args:
        trace (Trace): The Trace object containing the entries of the conversation.

        Returns:
        str: A formatted report as a string.
        """
        report = []
        answer = None
        document_quotes = []

        for entry in self.entries:
            if isinstance(entry, ToolResult):
                if isinstance(entry.output, Observation):
                    if entry.output.info_pieces:
                        for info_piece in entry.output.info_pieces:
                            if info_piece.quotable:
                                document_quotes.append(f"{info_piece.source}:\n{info_piece.text}\n\n---")
                elif isinstance(entry.output, Answer):
                    answer = entry.output

        report.append(f"User Question: {self.user_question}")
        report.append(f"Answer: {answer.normalized_answer()}")
        report.append(f"Reasoning: {answer.reasoning}")
        report.append("Analyzed Document Fragments:\n")
        report.extend(document_quotes)

        return "\n".join(report)


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


class LLMReactor:
    class LLMReactorError(Exception):
        pass

    def __init__(self, 
                model: str,
                toolbox: list[Callable|HasLLMTools],
                trace: Trace,
                max_llm_calls: int,
                client: object,
                reflection: Any,
                soft_reflection_validation=True,
                question_checks=None,
                case_insensitive=False,
                ):
        self.model = model
        self.toolbox = toolbox
        self.toolbox.append(self.finish)
        self.trace = trace
        self.max_llm_calls = max_llm_calls
        self.client = client
        self.soft_reflection_validation = soft_reflection_validation
        self.question_checks = [] if question_checks is None else question_checks
        self.case_insensitive = case_insensitive

        self.step = 0
        self.to_reflect = False
        self.finished = False
        self.answer = None
        self.soft_errors = []
        self.reflection_prompt = []
        self.what_have_we_learned = KnowledgeBase()
        self.no_tool_calls_message = None

    def get_tools(self) -> list[Callable]:
        if self.step == self.max_llm_calls + 1:
            toolbox = [self.finish]
        else:
            toolbox = self.toolbox
        tools = []
        for item in toolbox:
            if isinstance(item, HasLLMTools):
                tools.extend(item.get_llm_tools())
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
        return completion

    def query_and_process(self):
        schemas = get_tool_defs(self.get_tools())
        response = self.openai_query(self.trace.to_messages(), schemas)
        message = response.choices[0].message
        self.trace.add_entry(message)
        results = process_response(response, self.get_tools())
        if len(self.get_tools()) > 0 and len(results) == 0:
            self.soft_errors.append("No function call")
            self.to_reflect = False
            if self.no_tool_calls_message is not None:
                message = { 'role': 'assistant', 'content': self.no_tool_calls_message }
                self.trace.add_entry(message)
        for result in results:
            if result.error is not None:
                raise self.LLMReactorError(result.stack_trace)
            self.trace.add_entry(result)
            self.clean_context_reflection(result)
        return results

    def clean_context_reflection(self, result):
        # In clean context reflection we cannot ask the llm to plan - because it does not get the information retrieved previously.
        # But we can contrast the reflection with previous data ourselves - and for example remove links that were already retrieved.

        if not isinstance(result.output, Observation) or not result.output.reflection_needed():
            return

        learned_stuff = f"\n\nSo far we have some notes on the following urls:{self.what_have_we_learned.learned()}" if not self.what_have_we_learned.is_empty() else ""
        jump_next = f"\n- jump to the next occurrence of the looked up keyword" if result.name == 'lookup' or result.name == 'next' else ""
        sysprompt = f"""You are a researcher working on the following user question:
{self.trace.user_question}{learned_stuff}

You need to review the information retrieval recorded below."""

#. You need to decide if the next action should be:
#- read_more information from the current position on the current page
#- lookup a new keyword on the current page{jump_next}
#- go to a new url 
#- make a new wikipedia search
#- finish with an answer to the user question
#To save potential sources of new information please add their urls to the new_sources list."""
        user_prompt = str(result.output)
        messages = [
            {'role': 'system', 'content': sysprompt},
            {'role': 'user', 'content': user_prompt},
        ]
        schemas = get_tool_defs([ReflectionResult])
        response = self.openai_query(messages, schemas)
        message = response.choices[0].message
        #self.trace.add_entry(message)
        results = process_response(response, [ReflectionResult])
        if len(results) > 1:
            self.soft_errors.append(f"More than one reflection result")
        new_result = results[0]
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

        second_user_prompt = f"""What would you do next? 
You can choose from the following options:
{result.output.available_tools}
Specify the action and also its parameters.
Please explain your decision.
        """
        messages = [
            {'role': 'system', 'content': sysprompt},
            {'role': 'user', 'content': user_prompt},
            {'role': 'user', 'content': second_user_prompt},
        ]
        response = self.openai_query(messages, [])
        second_reflection = response.choices[0].message.content
        reflection_string = "**My Notes**\n" + reflection_string + "\n\n" + second_reflection
        message = { 'role': 'user', 'content': reflection_string }
        self.trace.add_entry(message)


    def process(self):
        self.analyze_question()
        while self.answer is None and self.step <= self.max_llm_calls + 1:
            self.query_and_process()
            self.step += 1
            if self.step == self.max_llm_calls:
                step_info = "\n\nThis was the last data you can get - in the next step you need to formulate your answer"
                self.trace.add_entry({'role': 'system', 'content': step_info})
            else:
                step_info = f"\n\nThis was {self.step} out of {self.max_llm_calls} calls for data."
                self.trace.add_entry({'role': 'system', 'content': step_info})

#            if 'gpt-4' in self.model:
#                time.sleep(20)


    def analyze_question(self):
        for query in self.question_checks:
            question_check = { 'role': 'user', 'content': query }
            logger.info(str(question_check))
            self.trace.add_entry(question_check)
            self.query_and_process()

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


def get_answer(question, config, client: object):
    print("\n\n<<< Question:", question)
    tool_class = config['tool']
    tool = tool_class(chunk_size=config['chunk_size'])
    toolbox = [tool]
    sys_prompt = PROMPTS[config['prompt_class']]
    initial_trace = Trace()
    initial_trace.add_system_message(sys_prompt(config['max_llm_calls'], ''))
    initial_trace.add_user_question(question)
    question_checks = QUESTION_CHECKS[config['question_check']] 
    reactor = LLMReactor(
        config['model'], toolbox, initial_trace, config['max_llm_calls'], client, None,
        question_checks=question_checks
    )
    reactor.process()
    return reactor

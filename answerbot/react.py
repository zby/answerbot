import json
import time
import logging
import copy
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Literal, Union, List, Dict, Annotated, Optional, Callable
from pprint import pprint
from dataclasses import dataclass

from .prompt_templates import QUESTION_CHECKS, PROMPTS, REFLECTIONS 

from llm_easy_tools import process_response, get_tool_defs, get_toolset_tools

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Get a logger for the current module
logger = logging.getLogger(__name__)

class Conversation:
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
            else:
                all_messages.append(entry.to_message())
        return all_messages



@dataclass(frozen=True)
class Reflection:
    reflection_class: Optional[str] = None
    message: Optional[str] = None
    detached: bool = True
    case_insensitive: bool = False

    def __post_init__(self):
        if (self.reflection_class is None) == (self.message is None):
            raise ValueError("reflection message and class cannot be used simultaneously")

    def prefix_class(self):
        if self.reflection_class and not self.detached:
            return self.reflection_class
        else:
            return None

    def prefix(self):
        if self.reflection_class and not self.detached:
            name = self.reflection_class.__name__
            if self.case_insensitive:
                name = name.lower()
            return f'{name}_and_'
        else:
            return ''


class LLMReactor:
    class LLMReactorError(Exception):
        pass

    def __init__(self, 
                 model: str,
                 toolbox: list[Callable],
                 conversation: Conversation,
                 max_llm_calls: int,
                 client: object,
                 reflection: Reflection,
                 soft_reflection_validation=True,
                 question_checks=None,
                 case_insensitive=False,
                 ):
        self.model = model
        self.toolbox = toolbox
        self.toolbox.append(self.finish)
        self.conversation = conversation
        self.max_llm_calls = max_llm_calls
        self.client = client
        self.soft_reflection_validation = soft_reflection_validation
        self.question_checks = [] if question_checks is None else question_checks
        self.case_insensitive = case_insensitive

        self.step = 0
        self.finished = False
        self.answer = None
        self.soft_errors = []
        if reflection.detached and reflection.reflection_class:
            self.toolbox.append(reflection.reflection_class)
        self.reflection = reflection

    def openai_query(self, tool_schemas, force_auto_tool_choice=False):
        args = {
            'model': self.model,
            'messages': self.conversation.to_messages()
        }
        if len(tool_schemas) == 0:
            pass
        elif len(tool_schemas) == 1 and not force_auto_tool_choice:
            args['tool_choice'] = {'type': 'function', 'function': {'name': tool_schemas[0]['function']['name']}}
            args['tools'] = tool_schemas
        else:
            args['tool_choice'] = "auto"
            args['tools'] = tool_schemas

        completion = self.client.chat.completions.create( **args )
        return completion


    def get_reflection(self):
        if self.reflection.reflection_class:
            tools = [self.reflection.reflection_class]
            self.query_and_process(tools)
        else:
            message = { 'role': 'user', 'content': self.reflection.message }
            logger.info(str(message))
            self.conversation.add_entry(message)
            self.query_and_process()


    def query_and_process(self, tools=[], additional_info='', no_tool_calls_message=None):
        schemas = get_tool_defs(tools, prefix_class=self.reflection.prefix_class())
        response = self.openai_query(schemas)
        message = response.choices[0].message.dict()
        if message['function_call'] is None:
            del message['function_call']
        if message['tool_calls'] is None:
            del message['tool_calls']
        self.conversation.add_entry(message)
        logger.info(str(message))
        results = process_response(response, tools, prefix_class=self.reflection.prefix_class())
        for result in results:
            if result.error is not None:
                raise self.LLMReactorError(result.error)
            message = result.to_message()
            message['content'] += additional_info
            logger.info(str(message))
            self.conversation.add_entry(message)
        if len(schemas) > 0 and len(results) == 0:
            self.soft_errors.append("No function call")
            if no_tool_calls_message is not None:
                message = { 'role': 'assistant', 'content': no_tool_calls_message }
                logger.info(str(message))
                self.conversation.add_entry(message)
        return results


    def process(self):
        self.analyze_question()
        while self.answer is None:
            self.step += 1
            if self.step == self.max_llm_calls + 1:
                tools = [self.finish]
            else:
                tools = self.toolbox
            if self.step == self.max_llm_calls:
                step_info = "\n\nThis was the last data you can get - in the next step you need to formulate your answer"
            else:
                step_info = f"\n\nThis was {self.step} out of {self.max_llm_calls} calls for data."
            no_tool_calls_message = "You did not ask for any data this time - but it still counts."
            if self.reflection.detached:
                self.get_reflection()
            self.query_and_process(tools, additional_info=step_info, no_tool_calls_message=no_tool_calls_message)

#            if 'gpt-4' in self.model:
#                time.sleep(20)


    def analyze_question(self):
        for query in self.question_checks:
            question_check = { 'role': 'user', 'content': query }
            logger.info(str(question_check))
            self.conversation.add_entry(question_check)
            self.query_and_process()

    def finish(self,
               answer: Annotated[str, "The answer to the user's question"],
               answer_short: Annotated[str, "A short version of the answer"],
               reasoning: Annotated[str, "The reasoning behind the answer. Think step by step. Mention all assumptions you make."],
               ambiguity: Annotated[Optional[str], "Have you found anything in the retrieved information that makes the question ambiguous? For example a search for some name can show that there are many different entities with the same name."] = None
    ):
        """
        Finish the task and return the answer.
        """
        self.answer=(
            self.normalize_answer(answer),
            self.normalize_answer(answer_short),
        )

    def normalize_answer(self, answer):
        answer = answer.strip(' \n.\'"')
        answer = answer.replace('’', "'")  # Replace all curly apostrophes with straight single quotes
        answer = answer.replace('"', "'")  # Replace all double quotes with straight single quotes
        if answer.lower() == 'yes' or answer.lower() == 'no':
            answer = answer.lower()
        return answer


def get_answer(question, config, client: object):
    print("\n\n<<< Question:", question)
    tool_class = config['tool']
    tool = tool_class(chunk_size=config['chunk_size'])
    toolbox = get_toolset_tools(tool)
    sys_prompt = PROMPTS[config['prompt_class']]
    reflection = Reflection(**REFLECTIONS[config['reflection']])
    initial_conversation = Conversation()
    initial_conversation.add_system_message(sys_prompt(config['max_llm_calls'], reflection.prefix()))
    initial_conversation.add_user_question(question)
    question_checks = QUESTION_CHECKS[config['question_check']] 
    reactor = LLMReactor(
        config['model'], toolbox, initial_conversation, config['max_llm_calls'], client, reflection,
        question_checks=question_checks
    )
    reactor.process()
    return reactor

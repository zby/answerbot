import json
import time
import logging
import copy
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Literal, Union, List, Dict
from pprint import pprint


from .prompt_templates import QUESTION_CHECKS, PROMPTS, REFLECTIONS 
from .tools_base import Finish

from llm_easy_tools import ToolBox

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Get a logger for the current module
logger = logging.getLogger(__name__)

class LLMReactor:
    class LLMReactorError(Exception):
        pass

    def __init__(self, model: str, toolbox: ToolBox,
                 conversation,
                 max_llm_calls: int, client,
                 reflection_class, reflection_message: str,
                 soft_reflection_validation=True,
                 reflection_detached=False,
                 question_checks=None,
                 ):
        self.model = model
        self.toolbox = toolbox
        self.conversation = conversation
        self.max_llm_calls = max_llm_calls
        self.client = client
        self.soft_reflection_validation = soft_reflection_validation
        self.reflection_class = reflection_class
        self.reflection_message = reflection_message
        self.question_checks = [] if question_checks is None else question_checks

        if self.reflection_message is not None and self.reflection_class is not None:
            raise ValueError("reflection message and class cannot be used simultaneously")

        self.step = 0
        self.finished = False
        self.answer = None
        self.soft_errors = []
        self.reflection_detached = reflection_detached
        if reflection_detached:
            self.toolbox.register_model(reflection_class)

    def openai_query(self, tool_schemas, force_auto_tool_choice=False):
        args = {
            'model': self.model,
            'messages': self.conversation
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

    def set_finished(self):
        self.finished = True



    def get_reflection(self):
        message = { 'role': 'user', 'content': self.reflection_message }
        logger.info(str(message))
        self.conversation.append(message)
        self.query_and_process()


    def query_and_process(self, schemas=[], additional_info='', no_tool_calls_message=None, prefix_class=None):
        response = self.openai_query(schemas)
        message = response.choices[0].message.dict()
        if message['function_call'] is None:
            del message['function_call']
        if message['tool_calls'] is None:
            del message['tool_calls']
        self.conversation.append(message)
        logger.info(str(message))
        results = self.toolbox.process_response(response)
        for result in results:
            if result.error is not None:
                if prefix_class:
                    self.soft_errors.append(result.error)
                    results = self.toolbox.process_response(response, prefix_class=prefix_class, ignore_prefix=True)
                    result = results[0]
                else:
                    raise self.LLMReactorError(result.error)
            if result.name == 'Finish':
                self.answer = result.model.normalized_answer()
                self.set_finished()
                return
            message = result.to_message()
            message['content'] += additional_info
            logger.info(str(message))
            self.conversation.append(message)
        if len(schemas) > 0 and len(results) == 0:
            self.soft_errors.append("No function call")
            if no_tool_calls_message is not None:
                message = { 'role': 'assistant', 'content': no_tool_calls_message }
                logger.info(str(message))
                self.conversation.append(message)
        return results


    def process(self):
        self.analyze_question()
        while not self.finished:
            self.step += 1
            if self.reflection_message:
                self.get_reflection()
                prefix_class = None
            elif self.reflection_detached:
                prefix_class = None
            else:
                prefix_class = self.reflection_class

            if self.reflection_detached and self.reflection_class:
                schema = self.toolbox.get_tool_schema(self.reflection_class.__name__)
                schemas = [schema]
                self.query_and_process(schemas, prefix_class=prefix_class)
            if self.step == self.max_llm_calls + 1:
                schemas = [self.toolbox.get_tool_schema('Finish', prefix_class)]  # Finish is registered
            else:
                schemas = self.toolbox.tool_schemas(prefix_class=prefix_class)
            #pprint(schemas)
            if self.step == self.max_llm_calls:
                step_info = "\n\nThis was the last data you can get - in the next step you need to formulate your answer"
            else:
                step_info = f"\n\nThis was {self.step} out of {self.max_llm_calls} calls for data."
            no_tool_calls_message = "You did not ask for any data this time - but it still counts."
            self.query_and_process(schemas, additional_info=step_info, no_tool_calls_message=no_tool_calls_message, prefix_class=prefix_class)

#            if 'gpt-4' in self.model:
#                time.sleep(20)


    def analyze_question(self):
        for query in self.question_checks:
            question_check = { 'role': 'user', 'content': query }
            logger.info(str(question_check))
            self.conversation.append(question_check)
            self.query_and_process()

def get_answer(question, config, client=None):
    print("\n\n<<< Question:", question)
    tool_class = config['tool']
    tool = tool_class(chunk_size=config['chunk_size'])
    toolbox = ToolBox()
    toolbox.register_toolset(tool)
    sys_prompt = PROMPTS[config['prompt_class']]
    reflection = REFLECTIONS[config['reflection']]
    reflection_class = None
    if 'class' in reflection:
        reflection_class = reflection['class']
    reflection_message = None
    if 'message' in reflection:
        reflection_message = reflection['message']
    reflection_detached = reflection.get('detached', False)
    if reflection_class and not reflection_detached:
        prefix = f'{reflection_class.__name__.lower()}_and_'
    else:
        prefix = ''
    initial_conversation = [
        { 'role': 'system', 'content': sys_prompt(config['max_llm_calls'], prefix) },
        { 'role': 'user', 'content': question }
    ]
    question_checks = QUESTION_CHECKS[config['question_check']] 
    reactor = LLMReactor(
        config['model'], toolbox, initial_conversation, config['max_llm_calls'], reflection_class=reflection_class,
        reflection_message=reflection_message, client=client, reflection_detached=reflection_detached,
        question_checks=question_checks
    )
    reactor.process()
    return reactor

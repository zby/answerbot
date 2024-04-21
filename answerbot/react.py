import httpx
import openai
import json
import time
import logging
import copy
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Literal, Union
from pprint import pprint

from dotenv import load_dotenv

from .prompt_builder import FunctionalPrompt, PromptMessage, Assistant, System, User, FunctionCall, FunctionResult
from .prompt_templates import QUESTION_CHECKS, PROMPTS, REFLECTIONS, AAEPrompt
from .tools_base import Finish

from llm_easy_tools import ToolBox

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Get a logger for the current module
logger = logging.getLogger(__name__)

# load OpenAI api key
load_dotenv()

class LLMReactor:
    def __init__(self, model: str, toolbox: ToolBox,
                 prompt: FunctionalPrompt,
                 max_llm_calls: int, client,
                 reflection_class, reflection_message: str,
                 soft_reflection_validation=True,
                 reflection_detached=False,
                 ):
        self.model = model
        self.toolbox = toolbox
        self.prompt = prompt
        self.max_llm_calls = max_llm_calls
        if client is None:
            client = openai.OpenAI(timeout=httpx.Timeout(70.0, read=60.0, write=20.0, connect=6.0))

        self.client = client
        self.soft_reflection_validation = soft_reflection_validation
        self.reflection_class = reflection_class
        self.reflection_message = reflection_message

        if self.reflection_message is not None and self.reflection_class is not None:
            raise ValueError("reflection message and class cannot be used simultaneously")

        self.step = 0
        self.finished = False
        self.answer = None
        self.reflection_errors = []
        self.reflection_detached = reflection_detached
        if reflection_detached:
            self.toolbox.register_model(reflection_class)

    def openai_query(self, tool_schemas, force_auto_tool_choice=False):
        args = {
            'model': self.model,
            'messages': self.prompt.to_messages()
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

    def message_from_response(self, response, choice_num=0, tool_num=0):
        #if response.choices[choice_num].message.function_call:
        #    function_call = response.choices[choice_num].message.function_call
        if response.choices[choice_num].message.tool_calls:
            function_call = response.choices[choice_num].message.tool_calls[tool_num].function
        else:
            function_call = None
        if function_call:
            print(function_call.arguments)
            function_args = json.loads(
                    function_call.arguments
                    .replace(', }', '}')
                    .replace(',}','}')
                    )
            message = FunctionCall(function_call.name, **function_args)
        else:
            message = Assistant(response.choices[choice_num].message.content)
        return message, function_call

    def get_reflection(self):
        message = User(self.reflection_message)
        logger.info(str(message))
        self.prompt.push(message)
        response = self.openai_query([])

        message, _ = self.message_from_response(response)
        self.prompt.push(message)
        logger.info(str(message))

    def process_prompt(self):
        while True:
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
                response = self.openai_query(schemas)
                message, _ = self.message_from_response(response)
                logger.info(str(message))                
                self.prompt.push(message)

            if self.step == self.max_llm_calls:
                schemas = [self.toolbox.get_tool_schema('Finish', prefix_class)]  # Finish is registered
            else:
                schemas = self.toolbox.tool_schemas(prefix_class=prefix_class)
            #pprint(schemas)
            response = self.openai_query(schemas)
            message, function_call = self.message_from_response(response)
            logger.info(str(message))
            self.prompt.push(message)
            if function_call is None:
                self.reflection_errors.append("No function call")
            else:

                result = self.toolbox.process_function(function_call, 'tool id', prefix_class=prefix_class)
                if result.error is not None and prefix_class is not None and self.soft_reflection_validation:
                    result = self.toolbox.process_function(function_call, 'tool_id', prefix_class=prefix_class, ignore_prefix=True)
                    self.reflection_errors.append(result.error)
                if isinstance(result.model, Finish):
                    #self.are_you_sure()
                    self.answer = result.model.normalized_answer()
                    self.set_finished()
                    return
            if self.step == self.max_llm_calls:
                self.set_finished()
                logger.info("<<< Max LLM calls reached without finishing")
                return

            if self.step == self.max_llm_calls - 1:
                step_info = "\n\nThis was the last data you can get - in the next step you need to formulate your answer"
            else:
                step_info = f"\n\nThis was {self.step} out of {self.max_llm_calls} calls for data."

            if function_call is not None:
                message = FunctionResult(function_call.name, result.output + f"\n\n{step_info}")
            else:
                message = Assistant("You did not ask for any data this time - but it still counts. " + step_info)
            logger.info(str(message))
            self.prompt.push(message)
#            if 'gpt-4' in self.model:
#                time.sleep(20)


    def analyze_question(self, queries):
        for query in queries:
            question_check = User(query)
            logger.info(str(question_check))
            self.prompt.push(question_check)
            response = self.openai_query([])
            message, _ = self.message_from_response(response)
            self.prompt.push(message)
            logger.info(str(message))

def get_answer_wiki(question, config, client=None):
    print("\n\n<<< Question:", question)
    tool_class = config['tool']
    tool = tool_class(chunk_size=config['chunk_size'])
    toolbox = ToolBox()
    toolbox.register_toolset(tool)
    prompt_class = PROMPTS[config['prompt_class']]
    reflection = REFLECTIONS[config['reflection']]
    reflection_class = None
    if 'class' in reflection:
        reflection_class = reflection['class']
    reflection_message = None
    if 'message' in reflection:
        reflection_message = reflection['message']
    reflection_detached = reflection.get('detached', False)
    prompt = prompt_class(question, config['max_llm_calls'], reflection_class, reflection_detached)
    reactor = LLMReactor(config['model'], toolbox, prompt, config['max_llm_calls'], reflection_class=reflection_class, reflection_message=reflection_message, client=client, reflection_detached=reflection_detached)
    reactor.analyze_question(QUESTION_CHECKS[config['question_check']])
    reactor.process_prompt()
    return reactor


get_answer = get_answer_wiki

import httpx
import openai
import json
import time
import logging
import copy
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Literal, Union
from pprint import pprint

from .prompt_builder import FunctionalPrompt, PromptMessage, Assistant, System, FunctionCall, FunctionResult
from .wikipedia_tool import WikipediaSearch

from llm_easy_tools import ToolBox

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Get a logger for the current module
logger = logging.getLogger(__name__)


class LLMReactor:
    def __init__(self, model: str, toolbox: ToolBox, prompt: FunctionalPrompt,
                 max_llm_calls: int, client, soft_reflection_validation=True):
        self.model = model
        self.toolbox = toolbox
        self.prompt = prompt
        self.max_llm_calls = max_llm_calls
        self.client = client
        self.soft_reflection_validation = soft_reflection_validation

        self.step = 0
        self.finished = False
        self.answer = None
        self.reflection_errors = []

    def openai_query(self, tool_schemas, force_auto_tool_choice=False):

        if len(tool_schemas) == 1 and not force_auto_tool_choice:
            tool_choice = tool_choice={'type': 'function', 'function': {'name': tool_schemas[0]['function']['name']}}
        else:
            tool_choice = "auto"

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.prompt.to_messages(),
            tools=tool_schemas,
            tool_choice=tool_choice
        )
        return completion

    def set_finished(self):
        self.finished = True

    def message_from_response(self, response, choice_num=0, tool_num=0):
        if response.choices[choice_num].message.function_call:
            function_call = response.choices[choice_num].message.function_call
        elif response.choices[choice_num].message.tool_calls:
            function_call = response.choices[choice_num].message.tool_calls[tool_num].function
        else:
            raise ValueError(f"Choice number {choice_num} in response is not a function nor tool call")
        function_args = json.loads(function_call.arguments)
        #pprint(function_args)
        message = FunctionCall(function_call.name, **function_args)
        return message, function_call


    def process_prompt(self):
        logger.debug(f"Processing prompt: {self.prompt}")
        self.step += 1
        if self.step == 1:
            prefix_class = None
        else:
            prefix_class = Reflection
        if self.step == self.max_llm_calls:
            schemas = [self.toolbox.get_tool_schema('Finish', prefix_class)]  # Finish is registered
        else:
            schemas = self.toolbox.tool_schemas(prefix_class=prefix_class)
        response = self.openai_query(schemas)
        message, function_call = self.message_from_response(response)
        logger.info(str(message))
        try:
            result = self.toolbox.process_function(function_call, prefix_class=prefix_class)
        except ValidationError as e:
            if prefix_class is not None and self.soft_reflection_validation:
                result = self.toolbox.process_function(function_call, prefix_class=prefix_class, ignore_prefix=True)
                self.reflection_errors.append(repr(e))
            else:
                raise e
        self.prompt.push(message)
        if isinstance(result, WikipediaSearch.Finish):
            self.answer = result.normalized_answer()
            self.set_finished()
            return
        elif self.step == self.max_llm_calls:
            self.set_finished()
            logger.info("<<< Max LLM calls reached without finishing")
            return

        if self.step == self.max_llm_calls - 1:
            step_info = "This was the last wikipedia result you can get - in the next step you need to formulate your answer"
        else:
            step_info = f"This was {self.step} out of {self.max_llm_calls} wikipedia calls."

        result = result + "\n\n" + step_info

        message = FunctionResult(function_call.name, result)
        logger.info(str(message))
        self.prompt.push(message)



class Reflection(BaseModel):
    how_relevant: Union[Literal[1, 2, 3, 4, 5], Literal['1', '2', '3', '4', '5']] = Field(
        ...,
        description="Was the last retrieved information relevant for answering this question? Choose 1, 2, 3, 4, or 5."
    )
    @field_validator('how_relevant')
    @classmethod
    def ensure_int(cls, v):
        if isinstance(v, str) and v in {'1', '2', '3', '4', '5'}:
            return int(v)  # Convert to int
        return v

    why_relevant: str = Field(..., description="Why the retrieved information was relevant?")
    next_actions_plan: str = Field(..., description="")


def get_answer(question, config):
    print("\n\n<<< Question:", question)
    wiki_search = WikipediaSearch(max_retries=2, chunk_size=config['chunk_size'])
    toolbox = ToolBox()
    toolbox.register_toolset(wiki_search)
    client = openai.OpenAI(timeout=httpx.Timeout(20.0, read=10.0, write=15.0, connect=4.0))
    reactor = LLMReactor(config['model'], toolbox, config['prompt'], config['max_llm_calls'], client=client)
    while True:
        print()
        print(f">>>LLM call number: {reactor.step}")
        reactor.process_prompt()
        # print(prompt.parts[-2])
        if reactor.finished:
            return reactor
#        if 'gpt-4' in config['model']:
#            time.sleep(59)

import httpx
import openai
import json
import time
import logging
import copy
from pydantic import BaseModel, Field, field_validator
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
                 max_llm_calls: int, client):
        self.model = model
        self.toolbox = toolbox
        self.prompt = prompt
        self.max_llm_calls = max_llm_calls
        self.client = client
        self.step = 0
        self.finished = False
        self.answer = None

    def openai_query(self, tool_schemas, tool_choice="auto"):
        args = {}
        args["tools"] = tool_schemas
        args["tool_choice"] = tool_choice

        response = None
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.prompt.to_messages(),
            **args
        )
        return completion

    def set_finished(self):
        self.finished = True

    def message_from_response(self, response, choice_num=0, tool_num=0):
        if response.choices[choice_num].message.function_call:
            function_call = response.choices[choice_num].message.function_call
        elif response.choices[choice_num].message.tool_calls:
            function_call = response.choices[choice_num].message.tool_calls[tool_num].function
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
            finish_schema = self.toolbox.get_tool_schema('Finish', prefix_class)
            response = self.openai_query([finish_schema],
                                         tool_choice={'type': 'function', 'function': {'name': 'Finish'}})
        else:
            all_schemas = self.toolbox.tool_schemas(prefix_class=prefix_class)
            response = self.openai_query(all_schemas)
        message, function_call = self.message_from_response(response)
        logger.info(str(message))
        result = self.toolbox.process_function(function_call, prefix_class=prefix_class)
        self.prompt.push(message)
        if isinstance(result, WikipediaSearch.Finish):
            self.answer = result.normalized_answer
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

import httpx
import openai
import json
import time
import logging
import copy
from pydantic import BaseModel, Field, validator
from typing import Literal

from .prompt_builder import FunctionalPrompt, PromptMessage, Assistant, System, FunctionCall, FunctionResult
from .get_wikipedia import WikipediaApi

from .react_prompt import FunctionalReactPrompt, NewFunctionalReactPrompt, TextReactPrompt
from .prompt_templates import NoExamplesReactPrompt
from .wikipedia_tool import WikipediaSearch

from llm_easy_tools import ToolBox

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Get a logger for the current module
logger = logging.getLogger(__name__)


class LLMReactor:
    def __init__(self, model: str, toolbox: ToolBox, prompt: FunctionalPrompt,
                 reflection_generator, max_llm_calls: int, client):
        self.model = model
        self.toolbox = toolbox
        self.prompt = prompt
        self.reflection_generator = reflection_generator
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
        message = FunctionCall(function_call.name, **function_args)
        return message, function_call.name


    def process_prompt(self):
        logger.debug(f"Processing prompt: {self.prompt}")
        self.step += 1
        finish_schema = self.toolbox.tool_registry["Finish"]["tool_schema"]
        all_schemas = self.toolbox.tool_schemas
        if self.step == self.max_llm_calls:
            response = self.openai_query([finish_schema],
                                         tool_choice={'type': 'function', 'function': {'name': 'Finish'}})
        else:
            response = self.openai_query(all_schemas)
        message, function_name = self.message_from_response(response)
        results = self.toolbox.process_response(response)
        result = results[0]
        logger.info(str(message))
        self.prompt.push(message)
        if result is WikipediaSearch.Finish:
            self.answer = result.normalized_answer
            self.set_finished()
            return
        elif self.step == self.max_llm_calls:
            self.set_finished()
            logger.info("<<< Max LLM calls reached without finishing")
            return

        message = FunctionResult(function_name, result)
        logger.info(str(message))
        self.prompt.push(message)


        message = self.reflection_generator.generate(self.step, self.max_llm_calls)
        logger.info(str(message))
        self.prompt.push(message)
        reflection_schema = self.toolbox.tool_registry["Reflection"]["tool_schema"]
        response = self.openai_query([reflection_schema],tool_choice={'type': 'function', 'function': {'name': 'Reflection'}})
        reflections = self.toolbox.process_response(response)
        relevant_score = reflections[0].how_relevant
        relevant_justification = reflections[0].why_relevant
        plan = reflections[0].next_actions_plan
        message = Assistant(f"On scale from 1 to 5 the last retrieved information revancy score is {relevant_score}.\n{relevant_justification}.\nNext action plan: {plan} ")
        logger.info(str(message))
        self.prompt.push(message)


class Reflection(BaseModel):
    how_relevant: Literal[1, 2, 3, 4, 5] = Field(
        ...,
        description="Was the last retrieved information relevant for answering this question? Choose 1, 2, 3, 4, or 5."
    )
    @validator('how_relevant', pre=True)
    def ensure_int(cls, v):
        if isinstance(v, str) and v in {'1', '2', '3', '4', '5'}:
            return int(v)  # Convert to int
        return v

    why_relevant: str = Field(..., description="Why the retrieved information was relevant?")
    next_actions_plan: str = Field(..., description="")


def get_answer(question, config):
    print("\n\n<<< Question:", question)
    wiki_api = WikipediaApi(max_retries=2, chunk_size=config['chunk_size'])
    wiki_search = WikipediaSearch(wiki_api)
    toolbox = ToolBox()
    toolbox.register_toolset(wiki_search)
    toolbox.register_tool(Reflection)
    toolbox.register_tool(WikipediaSearch.Finish)
    client = openai.OpenAI(timeout=httpx.Timeout(20.0, read=10.0, write=15.0, connect=4.0))
    reactor = LLMReactor(config['model'], toolbox, config['prompt'], config['reflection_generator'], config['max_llm_calls'], client=client)
    while True:
        print()
        print(f">>>LLM call number: {reactor.step}")
        reactor.process_prompt()
        # print(prompt.parts[-2])
        if reactor.finished:
            return reactor
#        if 'gpt-4' in config['model']:
#            time.sleep(59)

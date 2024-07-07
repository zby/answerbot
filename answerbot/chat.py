from typing import Literal, Union, Callable, Any, Protocol, Iterable, runtime_checkable, Optional, Annotated, Type, TypeVar
from dataclasses import dataclass, asdict, field, is_dataclass
from pydantic import BaseModel
from litellm import completion, ModelResponse
from jinja2 import Template
from dotenv import load_dotenv

from llm_easy_tools import get_tool_defs, process_response, ToolResult, LLMFunction 

from answerbot.tools.wiki_tool import WikipediaTool

import litellm
import logging
from pprint import pprint

load_dotenv()
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]
#litellm.success_callback=["helicone"]
#litellm.set_verbose=True


T = TypeVar('T')

def render(template_str: str, obj: T, chat: Optional[Any] = None) -> str:
    if not is_dataclass(obj):
        raise TypeError("The 'obj' parameter must be a dataclass instance")
    template = Template(template_str)
    fields = asdict(obj)
    return template.render(chat=chat, **fields)

@runtime_checkable
class HasLLMTools(Protocol):
    def get_llm_tools(self) -> Iterable[Callable]:
        pass

def expand_toolbox(toolbox: list[HasLLMTools|LLMFunction|Callable]) -> list[Callable|LLMFunction]:
    tools = []
    for item in toolbox:
        if isinstance(item, HasLLMTools):
            tools.extend(item.get_llm_tools())
        else:
            tools.append(item)
    return tools

@dataclass(frozen=True)
class Prompt:
    pass

@dataclass(frozen=True)
class Question(Prompt):
    question: str
    max_llm_calls: int

@dataclass
class Chat:
    entries: list[Prompt|ToolResult|dict]
    templates: dict[Type[Prompt], str] = field(default_factory=dict)

    def make_message(self, prompt: Prompt, role: str = 'user') -> str:
        template_str = self.templates[type(prompt)]
        content = render(template_str, prompt, self)
        return {
            'role': role,
            'content': content.strip()
        }

    def to_messages(self) -> list[dict]:
        messages = []
        for message in self.entries:
            if isinstance(message, Prompt):
                messages.append(self.make_message(message))
            elif isinstance(message, ToolResult):
                messages.append(message.to_message())
            else:
                messages.append(message)
        return messages

    def user_question(self) -> Optional[str]:
        for entry in self.entries:
            if isinstance(entry, Question):
                return entry.question
        return None

@dataclass
class ChatProcessor:
    model: str
    one_tool_per_step: bool = True
    # With statefull tools with many tool calls the LLM gets confused about the state of the tools
    # There should be an option in the litellm api for that

    def llm_reply(self, chat: Chat, schemas=[]) -> ModelResponse:
        messages = chat.to_messages()
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

        result = completion(**args)
        message = result.choices[0].message

        if self.one_tool_per_step and hasattr(message, 'tool_calls') and message.tool_calls:
            #print(f"Tool calls: {result.choices[0].message.tool_calls}")
            if len(message.tool_calls) > 1:
                logging.warning(f"More than one tool call: {message.tool_calls}")
                message.tool_calls = [message.tool_calls[0]]

        if len(schemas) > 0:
            if not hasattr(message, 'tool_calls') or not message.tool_calls:
                logging.warning(f"No function call:\n")

        chat.entries.append(message)

        return result

    def process(self, chat: Chat, tools: list[Callable|LLMFunction]):
        schemas = get_tool_defs(tools)
        response = self.llm_reply(chat, schemas)
        results = process_response(response, tools)
        for result in results:
            chat.entries.append(result)

@dataclass
class Answer:
    """
    Answer to the question.
    """
    answer: str
    reasoning: str

@dataclass(frozen=True)
class StepInfo(Prompt):
    step: int
    max_steps: int

@dataclass
class QAApp:
    toolbox: list[HasLLMTools|LLMFunction|Callable]
    max_iterations: int
    model: str
    system_prompt: str
    user_prompt_template: str
    answer_template: str
    step_info_template: str

    def __post_init__(self):
        self.step = 0
        self._chat_processor = ChatProcessor(model=self.model, one_tool_per_step=True)
        self.chat = Chat(
            templates = {
                Prompt: self.system_prompt,
                Question: self.user_prompt_template,
                Answer: self.answer_template,
                StepInfo: self.step_info_template
            },
            entries=[Prompt()])

    def get_tools(self) -> list[Callable|LLMFunction]:
        tools = [Answer]
        if self.step < self.max_iterations:
            tools.extend(expand_toolbox(self.toolbox))
        return tools

    def process(self, question: str):
        chat = self.chat
        chat.entries.append(Question(question, self.max_iterations))
        while(self.step <= self.max_iterations):
            tools = self.get_tools()
            self._chat_processor.process(chat, tools)
            result = chat.entries[-1]
            if result.error:
                raise Exception(result.error)
            if result.soft_errors:
                for soft_error in result.soft_errors:
                    logging.warning(soft_error)
            if isinstance(result.output, Answer):
                answer = result.output
                return render(self.answer_template, answer, chat)
            chat.entries.append(StepInfo(self.step, self.max_iterations))
            self.step += 1
        return None

sys_prompt = """You are a helpful assistant with extensive knowledge of wikipedia.
You always try to support your answer with quotes from wikipedia.
You remember that the information you receive from the wikipedia api is not the full page - it is just a fragment.
You always try to answer the user question, even if it is ambiguous, just note the necessary assumptions.
You Work carefully - never make two calls to wikipedia in the same step."""

user_prompt_template = """Please answer the following question. You can use wikipedia for reference - but think carefully about what pages exist at wikipedia.
You have only {{max_llm_calls}} calls to the wikipedia API.
When searching wikipedia never make any complex queries, always decide what is the main topic you are searching for and put it in the search query.
When you want to know a property of an object or person - first find the page of that object or person and then browse it to find the property you need.

When you know the answer call Answer. Please make the answer as short as possible. If it can be answered with yes or no that is best.
Remove all explanations from the answer and put them into the reasoning field.

Question: {{question}}"""

answer_template = '''The answer to the question:"{{chat.user_question()}}" is:
{{ answer }}

Reasoning:
{{ reasoning }}'''

step_info_template = '''
Step: {{step + 1}} of {{max_steps + 1}}
{% if step >= max_steps - 1 %}
This was the last data retrieval in the next step you must provide an answer to the user question
{% endif %}
'''

model='claude-3-5-sonnet-20240620'
#model='gpt-3.5-turbo'

app = QAApp(
    toolbox=[WikipediaTool(chunk_size=400)],
    max_iterations=5,
    model=model,
    system_prompt=sys_prompt,
    user_prompt_template=user_prompt_template,
    answer_template=answer_template,
    step_info_template=step_info_template
)

question = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
print(app.process(question))
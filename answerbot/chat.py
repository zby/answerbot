from typing import Literal, Union, Callable, Any, Protocol, Iterable, runtime_checkable, Optional, Annotated, Type, TypeVar
from dataclasses import dataclass, asdict, field, is_dataclass
from pydantic import BaseModel
from litellm import completion, ModelResponse
from jinja2 import Template

from llm_easy_tools import get_tool_defs, process_response, ToolResult, LLMFunction 

from answerbot.tools.wiki_tool import WikipediaTool

import logging

T = TypeVar('T')

def render_prompt(template_str: str, obj: T, context: Optional[object] = None) -> str:
    if not is_dataclass(obj):
        raise TypeError("The 'obj' parameter must be a dataclass instance")
    template = Template(template_str)
    fields = asdict(obj)
    result = template.render(context=context, **fields)
    return result

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

    @classmethod
    def template(cls) -> str:
        raise NotImplementedError("Subclasses must implement this method")

@dataclass
class Chat:
    model: str
    entries: list[Prompt|ToolResult|dict] = field(default_factory=list)
    templates: dict[Type[Prompt], str] = field(default_factory=dict)
    system_prompt: Optional[Prompt] = None
    one_tool_per_step: bool = True
    # With statefull tools with many tool calls the LLM gets confused about the state of the tools
    # There should be an option in the litellm api for that
    context: object = None
    # for use in prompt templates


    def make_message(self, prompt: Prompt, role: str = 'user') -> str:
        template_str = prompt.template()
        content = render_prompt(template_str, prompt, self.context)
        return {
            'role': role,
            'content': content.strip()
        }

    def to_messages(self) -> list[dict]:
        if self.system_prompt:
            messages = [self.make_message(self.system_prompt, 'system')]
        else:
            messages = []
        for message in self.entries:
            if isinstance(message, Prompt):
                messages.append(self.make_message(message))
            elif isinstance(message, ToolResult):
                messages.append(message.to_message())
            else:
                messages.append(message)
        return messages

    def llm_reply(self, schemas=[]) -> ModelResponse:
        messages = self.to_messages()
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

        self.entries.append(message)

        return result

    def process(self, context: object, tools: list[Callable|LLMFunction]):
        schemas = get_tool_defs(tools)
        response = self.llm_reply(schemas)
        results = process_response(response, tools)
        for result in results:
            self.entries.append(result)
        return results


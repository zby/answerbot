from typing import Literal, Union, Callable, Any, Protocol, Iterable, runtime_checkable, Optional, Annotated
from dataclasses import dataclass, asdict
from pydantic import BaseModel
from litellm import completion, ModelResponse

from llm_easy_tools import get_tool_defs, process_response, ToolResult

import logging

@runtime_checkable
class HasLLMTools(Protocol):
    def get_llm_tools(self) -> Iterable[Callable]:
        pass


@dataclass(frozen=True)
class Prompt:
    prompt_template: str

    def to_message(self) -> dict:
        # Create a dictionary of all fields, excluding prompt_template
        fields = {k: v for k, v in asdict(self).items() if k != 'prompt_template'}
        content = self.prompt_template.format(**fields)
        return {
            'role': 'user',
            'content': content.strip()
        }

@dataclass(frozen=True)
class Question(Prompt):
    question: str
    max_llm_calls: int

@dataclass
class Chat:
    messages: list[Prompt|dict]

    def to_messages(self) -> list[dict]:
        messages = []
        for message in self.messages:
            if isinstance(message, Prompt):
                messages.append(message.to_message())
            else:
                messages.append(message)
        return messages
    
    def user_question(self) -> Optional[str]:
        for entry in self.entries:
            if isinstance(entry, Question):
                return entry.question
        return None

class ChatProcessor(BaseModel):
    model: str
    one_tool_per_step: bool = True

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

        # Remove all but one tool_call from the result if present
        # With many tool calls the LLM gets confused about the state of the tools
        if self.one_tool_per_step and hasattr(message, 'tool_calls') and message.tool_calls:
            #print(f"Tool calls: {result.choices[0].message.tool_calls}")
            if len(message.tool_calls) > 1:
                logging.warning(f"More than one tool call: {message.tool_calls}")
                message.tool_calls = [message.tool_calls[0]]

        if len(schemas) > 0:
            if not hasattr(message, 'tool_calls') or not message.tool_calls:
                logging.warning(f"No function call:\n")

        self.append(message)

        return result

    def process_tools(self, chat: Chat, tools: list[Callable]):
        schemas = get_tool_defs(tools)
        response = self.llm_reply(chat, schemas)
        results = process_response(response, tools)
        for result in results:
            chat.append(result.to_message())

@dataclass
class ToolLoop:
    chat_processor: ChatProcessor
    toolbox: list[HasLLMTools|Callable]
    max_iterations: int = 5
    iteration: int = 0
    result: Any = None

    def last_step(self):
        return self.iteration >= self.max_iterations

    def process(self, chat: Chat):
        while not self.last_step():
            self.iteration += 1
            result = self.do_one_step(chat)
            if result is not None:
                return result

    def do_one_step(self, chat: Chat):
        tools = self.get_tools()
        results = self.chat_processor.process_tools(chat, tools)
        for result in results:
            if result.name == 'finish':
                return result

    def get_tools(self) -> list[Callable]:
        tools = [self.finish]
        if not self.last_step():
            for item in self.toolbox:
                if isinstance(item, HasLLMTools):
                    new_tools = item.get_llm_tools()
                    tools.extend(new_tools)
                else:
                    tools.append(item)
        return tools

    def finish(self, result):
        """
        Finish the task and return the answer.
        """
        return result

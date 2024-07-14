from typing import Callable, Protocol, Iterable, runtime_checkable, Optional, Type
from dataclasses import dataclass, field
from litellm import completion, ModelResponse, Message
from jinja2 import Template, Environment, ChoiceLoader, FileSystemLoader, DictLoader, BaseLoader
from pprint import pformat

from llm_easy_tools import get_tool_defs, process_response, ToolResult, LLMFunction 

import logging


# Configure logging for this module
logger = logging.getLogger('answerbot.chat')

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
    def role(self) -> str:
        return 'user'

@dataclass(frozen=True)
class SystemPrompt(Prompt):
    """
    System prompt for the chat.
    """
    def role(self) -> str:
        return 'system'


class StringLoader(BaseLoader):
    def __init__(self, template_string):
        self.template_string = template_string

    def get_source(self, environment, template):
        # Always return the template string, regardless of the requested name
        return self.template_string, None, lambda: True

@dataclass
class TemplateManager:
    templates: dict[str, str] = field(default_factory=dict)
    templates_dirs: list[str] = field(default_factory=list)
    fallback_template: str = "{{ __str__() }}"  # Default fallback template using __str__

    def __post_init__(self):
        self.env = self._create_environment()

    def _create_environment(self) -> Environment:
        loaders = [
            DictLoader(self.templates),
            FileSystemLoader(self.templates_dirs),
            StringLoader(self.fallback_template)
        ]
        return Environment(loader=ChoiceLoader(loaders))

    def render_prompt(self, obj: object, context: Optional[object] = None) -> str:
        template_name = type(obj).__name__
        template = self.env.get_template(template_name)

        # Create a context dictionary with the object's attributes and methods
        obj_context = {name: getattr(obj, name) for name in dir(obj)}

        # Merge with the provided context, if any
        if context:
            obj_context.update({'c': context})

        result = template.render(**obj_context)
        return result

@dataclass
class Chat:
    model: str
    messages: list[dict|Message] = field(default_factory=list)
    template_manager: Optional[TemplateManager] = None
    templates: dict[str, str] = field(default_factory=dict)    # when you don't want to create a template manager yourself
    templates_dirs: list[str] = field(default_factory=list)    # ^^^^
    system_prompt: Optional[Prompt] = None
    context: Optional[object] = None # passed as 'c' to the template
    metadata: Optional[dict] = None  # passed to completion() - I use it for observability (tagging traces in langfuse)
    fail_on_tool_error: bool = True  # if False the error message is passed to the LLM to fix the call, if True exception is raised
    one_tool_per_step: bool = True  # for stateful tools executing more than one tool call per step is often confusing for the LLM

    def __post_init__(self):
        if self.template_manager and (self.templates or self.templates_dirs):
            raise ValueError("Cannot specify both template_manager and templates/templates_dirs")

        if self.templates or self.templates_dirs:
            self.template_manager = TemplateManager(
                templates=self.templates,
                templates_dirs=self.templates_dirs
            )
        elif self.template_manager is None:
            self.template_manager = TemplateManager()

        if self.system_prompt:
            system_message = self.make_message(self.system_prompt)
            self.messages.insert(0, system_message)

    def make_message(self, prompt: object) -> dict:
        if hasattr(prompt, 'c'):
            raise ValueError("Prompt object cannot have an attribute named 'c' as it conflicts with the context parameter in render_prompt.")
        content = self.template_manager.render_prompt(prompt, self.context)
        role = getattr(prompt, 'role', 'user')
        return {
            'role': role,
            'content': content.strip()  #TODO: is .strip() needed here?
        }

    def append(self, message: Prompt|ToolResult|dict|Message):
        if isinstance(message, Prompt):
            message_dict = self.make_message(message)
        elif isinstance(message, ToolResult):
            message_dict = message.to_message()
        elif isinstance(message, dict) or isinstance(message, Message):
            message_dict = message
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
        self.messages.append(message_dict)

    def llm_reply(self, schemas=[]) -> ModelResponse:
        args = {
            'model': self.model,
            'messages': self.messages
        }
        if self.metadata:
            args['metadata'] = self.metadata

        if len(schemas) > 0:
            args['tools'] = schemas
            if len(schemas) == 1:
                args['tool_choice'] = {'type': 'function', 'function': {'name': schemas[0]['function']['name']}}
            else:
                args['tool_choice'] = "auto"

        logger.debug(f"llm_reply args: {pformat(args)}")

        result = completion(**args)
        message = result.choices[0].message

        if self.one_tool_per_step and hasattr(message, 'tool_calls') and message.tool_calls:
            if len(message.tool_calls) > 1:
                logging.warning(f"More than one tool call: {message.tool_calls}")
                message.tool_calls = [message.tool_calls[0]]

        if len(schemas) > 0:
            if not hasattr(message, 'tool_calls') or not message.tool_calls:
                logging.warning(f"No function call:\n")

        self.append(message)

        return result

    def process(self, toolbox: list[HasLLMTools|LLMFunction|Callable]):
        tools = expand_toolbox(toolbox)
        schemas = get_tool_defs(tools)
        response = self.llm_reply(schemas)
        results = process_response(response, tools)
        outputs = []
        for result in results:
            if result.soft_errors:
                for soft_error in result.soft_errors:
                    logger.warning(soft_error)
            self.append(result)
            if result.error and self.fail_on_tool_error:
                raise Exception(result.error)
            outputs.append(result.output)

        return outputs


if __name__ == "__main__":

    @dataclass(frozen=True)
    class AssistantPrompt(Prompt):
        answer: str

        def role(self) -> str:
            return 'assistant'

    @dataclass(frozen=True)
    class SpecialPrompt(Prompt):
        content: str

        def __str__(self):
            return f"Special prompt: {self.content.upper()}"

    @dataclass(frozen=True)
    class Prompt1(Prompt):
        value: str

    @dataclass(frozen=True)
    class Prompt2(Prompt):
        value: str

    # create Chat with default template manager
    chat = Chat(
        model="gpt-3.5-turbo",
        system_prompt=SystemPrompt(),
        templates_dirs=["tests/data/prompts1", "tests/data/prompts2"],
        templates={
            "SystemPrompt": "You are a helpful assistant.",
            "AssistantPrompt": "Assistant: {{answer}}",
            "SpecialPrompt": "{{__str__()}}"
        }
    )

    # Create example prompts
    prompt1_from_prompts1 = Prompt1(value="Example1")
    prompt2 = Prompt2(value="Example2")
    assistant_prompt = AssistantPrompt(answer="This is an assistant response.")
    special_prompt = SpecialPrompt(content="This is a special message")

    # Add prompts to the chat
    chat.append(prompt1_from_prompts1)
    chat.append(prompt2)
    chat.append(assistant_prompt)
    chat.append(special_prompt)

    # Print out entries from the Chat
    print("Chat entries:")
    for i, message in enumerate(chat.messages):
        print(f"{i + 1}. {message['role']}: {message['content']}")

    chat = Chat(
        model="gpt-3.5-turbo",
        system_prompt=SystemPrompt(),
        templates_dirs=["tests/data/prompts2"],
        templates={
            "SystemPrompt": "You are a helpful assistant.",
            "AssistantPrompt": "Assistant: {{answer}}",
            "SpecialPrompt": "{{__str__()}}"
        }
    )
    # Create example prompts
    prompt1_from_prompts2 = Prompt1(value="Example1")
    # Add prompts to the chat
    chat.append(prompt1_from_prompts2)

    # Print out entries from the Chat
    print("Chat entries:")
    for i, message in enumerate(chat.messages):
        print(f"{i + 1}. {message['role']}: {message['content']}")

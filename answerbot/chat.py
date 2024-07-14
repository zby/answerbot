from typing import Callable, Protocol, Iterable, runtime_checkable, Optional, Type
from dataclasses import dataclass, field
from litellm import completion, ModelResponse, Message
from jinja2 import Template
from pprint import pformat

from llm_easy_tools import get_tool_defs, process_response, ToolResult, LLMFunction 

import logging
import os


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

@dataclass
class Chat:
    model: str
    messages: list[dict|Message] = field(default_factory=list)
    templates: dict[str, str] = field(default_factory=dict)
    templates_dirs: list[str] = field(default_factory=list)
    system_prompt: Optional[Prompt] = None
    one_tool_per_step: bool = True
    # With statefull tools with many tool calls the LLM gets confused about the state of the tools
    # There should be an option in the litellm api for that
    # TODO: Add a fork of the process if that happens - and then collect all resolutions from all forks at the end (like with an Non Deterministic Finit Automaton)
    context: Optional[object] = None  # for use in prompt templates
    metadata: Optional[dict] = None   # passed to litellm completion - useful for tagging prompts in langfuse
    fail_on_tool_error: bool = True

    def __post_init__(self):
        if self.system_prompt:
            system_message = self.make_message(self.system_prompt)
            self.messages.insert(0, system_message)
        templates_in_files = self.load_templates(self.templates_dirs)
        templates_in_files.update(self.templates)
        self.templates = templates_in_files

    def load_templates(self, templates_dirs: list[str]) -> dict[str, str]:
        templates_in_files = {}
        for templates_dir in templates_dirs:
            for filename in os.listdir(templates_dir):
                class_name, file_type = os.path.splitext(filename)
                if file_type == '.jinja2':
                    class_name = os.path.basename(class_name)  # Remove any path components
                    if class_name not in templates_in_files:  # Only load if not already found
                        with open(os.path.join(templates_dir, filename), 'r') as file:
                            templates_in_files[class_name] = file.read()
        return templates_in_files

    def render_prompt(self, obj: object, context: Optional[object] = None) -> str:
        template_str = self.templates[type(obj).__name__]
        template = Template(template_str)
        fields = {name: value for name, value in obj.__dict__.items()}
        fields.update({name: getattr(obj, name) for name in dir(obj) if isinstance(getattr(type(obj), name, None), property)})
        result = template.render(c=context, **fields)
        return result


    def make_message(self, prompt: Prompt) -> str:
        # Check if prompt has an attribute 'c'
        if hasattr(prompt, 'c'):
            raise ValueError("Prompt object cannot have an attribute named 'c' as it conflicts with the context parameter in render_prompt.")
        content = self.render_prompt(prompt, self.context)
        return {
            'role': prompt.role(),
            'content': content.strip()
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

        logger.debug(f"Args: {pformat(args)}")

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
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class SystemPrompt(Prompt):
        pass

    @dataclass(frozen=True)
    class UserPrompt(Prompt):
        question: str

    @dataclass(frozen=True)
    class AssistantPrompt(Prompt):
        answer: str

    # Create example prompts
    system_prompt = SystemPrompt()
    user_prompt = UserPrompt(question="What is the capital of France?")
    assistant_prompt = AssistantPrompt(answer="The capital of France is Paris.")
    tool_result = ToolResult(
        tool_call_id="123",
        name="population_lookup",
        output="The population of Paris is 2,148,276 inhabitants."
    )


    # Create a Chat instance
    chat = Chat(
        model="gpt-3.5-turbo",
        system_prompt=system_prompt,
        templates={
            SystemPrompt: "You are a helpful assistant.",
            UserPrompt: "User: {{question}}",
            AssistantPrompt: "Assistant: {{answer}}"
        }
    )

    # Add prompts to the chat
    chat.append(user_prompt)
    chat.append(assistant_prompt, 'ASSISTANT')
    chat.append(tool_result)

    # Print out entries from the Chat
    print("Chat entries:")
    for i, message in enumerate(chat.messages):
        print(f"{i + 1}. {message['role']}: {message['content']}")
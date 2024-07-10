from typing import Callable, Protocol, Iterable, runtime_checkable, Optional, Type
from dataclasses import dataclass, field
from litellm import completion, ModelResponse, Message
from jinja2 import Template

from llm_easy_tools import get_tool_defs, process_response, ToolResult, LLMFunction 

import logging

# Configure logging for this module
logger = logging.getLogger('anserbot.chat')

def render_prompt(template_str: str, obj: object, context: Optional[object] = None) -> str:
    template = Template(template_str)
    fields = {field.name: getattr(obj, field.name) for field in obj.__dataclass_fields__.values()}
    result = template.render(c=context, **fields)
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
    pass

@dataclass(frozen=True)
class SystemPrompt(Prompt):
    """
    System prompt for the chat.
    """
    pass

@dataclass
class Chat:
    model: str
    messages: list[dict|Message] = field(default_factory=list)
    templates: dict[Type[Prompt], str] = field(default_factory=dict)
    system_prompt: Optional[Prompt] = None
    one_tool_per_step: bool = True
    # With statefull tools with many tool calls the LLM gets confused about the state of the tools
    # There should be an option in the litellm api for that
    # TODO: Add a fork of the process if that happens - and then collect all resolutions from all forks at the end (like with an Non Deterministic Finit Automaton)
    context: Optional[object] = None
    # for use in prompt templates

    def __post_init__(self):
        if self.system_prompt:
            system_message = self.make_message(self.system_prompt, 'system')
            self.messages.insert(0, system_message)


    def make_message(self, prompt: Prompt, role: str = 'user') -> str:
        # Check if prompt has an attribute 'c'
        if hasattr(prompt, 'c'):
            raise ValueError("Prompt object cannot have an attribute named 'c' as it conflicts with the context parameter in render_prompt.")
        template_str = self.templates[type(prompt)]
        content = render_prompt(template_str, prompt, self.context)
        return {
            'role': role,
            'content': content.strip()
        }

    def append(self, message: Prompt|ToolResult|dict|Message, role: Optional[str] = None):
        if isinstance(message, Prompt):
            message_dict = self.make_message(message)
        elif isinstance(message, ToolResult):
            message_dict = message.to_message()
        elif isinstance(message, dict) or isinstance(message, Message):
            message_dict = message
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
        if role:
            message_dict['role'] = role
        self.messages.append(message_dict)

    def llm_reply(self, schemas=[]) -> ModelResponse:
        args = {
            'model': self.model,
            'messages': self.messages
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
            if result.error:
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


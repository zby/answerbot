from typing import Callable, Optional, Type
from dataclasses import dataclass, field
from litellm import completion, ModelResponse, Message
from jinja2 import Environment, ChoiceLoader, FileSystemLoader, DictLoader, BaseLoader
from pprint import pformat

from llm_easy_tools import get_tool_defs, LLMFunction
from llm_easy_tools.processor import process_message

import logging


# Configure logging for this module
logger = logging.getLogger('answerbot.chat')
logger.setLevel(logging.DEBUG)  # Set the logger to capture DEBUG level messages


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
    template_env: Optional[Environment] = None
    templates: dict[str, str] = field(default_factory=dict)
    templates_dirs: list[str] = field(default_factory=list)
    system_prompt: Optional[Prompt] = None
    fail_on_tool_error: bool = True  # if False the error message is passed to the LLM to fix the call, if True exception is raised
    one_tool_per_step: bool = True  # for stateful tools executing more than one tool call per step is often confusing for the LLM
    saved_tools: list[LLMFunction|Callable] = field(default_factory=list)
    retries: int = 3

    def __post_init__(self):
        if self.template_env and (self.templates or self.templates_dirs):
            raise ValueError("Cannot specify both template_env and templates/templates_dirs")
        if not self.template_env and not self.templates and not self.templates_dirs:
            logger.warning("No template environment specified and no parameters for it, using empty environment")
        # TODO this needs a better solution
        #    raise ValueError("Must specify either template_env or templates/templates_dirs")

        if not self.template_env:
            self.template_env = self._create_environment()

        if self.system_prompt:
            if isinstance(self.system_prompt, str):
                system_prompt = {'role': 'system', 'content': self.system_prompt}  # the default role is 'user'
            else:
                system_prompt = self.system_prompt
            self.append(system_prompt)

    def _create_environment(self) -> Environment:
        loaders = [
            DictLoader(self.templates),
            FileSystemLoader(self.templates_dirs),
        ]
        return Environment(loader=ChoiceLoader(loaders))

    def render_prompt(self, obj: object, **kwargs) -> str:
        template_name = type(obj).__name__
        template = self.template_env.get_template(template_name)

        # Create a context dictionary with the object's public attributes and methods
        obj_context = {name: getattr(obj, name) for name in dir(obj) if not name.startswith('_')}

        # Merge with kwargs
        obj_context.update(kwargs)

        result = template.render(**obj_context)
        return result

    def make_message(self, prompt: Prompt) -> dict:
        content = self.render_prompt(prompt)
        return {
            'role': prompt.role(),
            'content': content.strip()
        }

    def __call__(self, message: Prompt|dict|Message|str, **kwargs) -> str:
        """
        Allow the Chat object to be called as a function.
        Appends the given message and calls llm_reply with the provided kwargs.
        Returns the content of the response message as a string.
        """
        self.append(message)
        response = self.llm_reply(**kwargs)
        return response.choices[0].message.content


    def append(self, message: Prompt|dict|Message|str) -> None:
        """
        Append a message to the chat.
        """
        if isinstance(message, Prompt):
            message_dict = self.make_message(message)
        elif isinstance(message, dict) or isinstance(message, Message):
            message_dict = message
        elif isinstance(message, str):
            message_dict = {'role': 'user', 'content': message}
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
        self.messages.append(message_dict)

    def llm_reply(self, tools=[], **kwargs) -> ModelResponse:
        self.saved_tools = tools
        schemas = get_tool_defs(tools)
        args = {
            'model': self.model,
            'messages': self.messages,
            'num_retries': self.retries
        }

        if len(schemas) > 0:
            args['tools'] = schemas
            if len(schemas) == 1:
                args['tool_choice'] = {'type': 'function', 'function': {'name': schemas[0]['function']['name']}}
            else:
                args['tool_choice'] = "auto"

        args.update(kwargs)

        logger.debug(f"llm_reply args: {pformat(args)}")
        logger.debug(f"Sending request to LLM with {len(self.messages)} messages")

        result = completion(**args)

        logger.debug(f"Received response from LLM: {result}")

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

    def process(self, **kwargs):
        if not self.messages:
            raise ValueError("No messages to process")
        message = self.messages[-1]
        results = process_message(message, self.saved_tools, **kwargs)
        outputs = []
        for result in results:
            if result.soft_errors:
                for soft_error in result.soft_errors:
                    logger.warning(soft_error)
            self.append(result.to_message())
            if result.error and self.fail_on_tool_error:
                print(result.stack_trace)
                raise Exception(result.error)
            if isinstance(result.output, Prompt):
                output = self.render_prompt(result.output)
                outputs.append(output)
            else:
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

        def render(self):
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

    # Add prompts to the chat
    chat.append(prompt1_from_prompts1)
    chat.append(prompt2)
    chat.append(assistant_prompt)

    # Print out entries from the Chat
    print("Chat entries:")
    for i, message in enumerate(chat.messages):
        print(f"{i + 1}. {message['role']}: {message['content']}")

    # This does ot work!!!
#    @dataclass(frozen=True)
#    class TestPrompt(Prompt):
#        role: str
#
#    test_prompt = TestPrompt(role="some role")
#    try:
#        chat.make_message(test_prompt)
#    except ValueError as e:
#        print(f"Error message: {str(e)}")
#
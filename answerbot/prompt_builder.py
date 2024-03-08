import json
from string import Template
from typing import Any, Iterable, List, Dict
from pprint import pformat

class PromptMessage:
    def __init__(self, content, **kwargs: dict):
        for key, value in self.defaults().items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.content = content

    def set_template_args(self, args):
        self.template_args = args

    def get_content(self) -> str:
        if self.summarized_below:
            content = "This message is summarized in subsequent messages."
        elif hasattr(self, 'template_args'):
            template = Template(self.content)
            content = template.substitute(self.template_args)
        else:
            content = self.content
        return content

    def openai_message(self) -> dict:
        role = self.role
        content = self.get_content()
        return {
            "role": role,
            "content": content,
        }

    def plaintext(self) -> str:
        return self.get_content()

    @classmethod
    def defaults(cls) -> dict:
        return {
            "role": cls.__name__.lower(),
            "summarized_below": False,
        }

    @classmethod
    def positional_arguments(cls) -> List[str]:
        return ['content']

    def __repr__(self):
        attrs = self.__dict__.copy()
        for key in self.defaults():
            if key in attrs and attrs[key] == self.defaults()[key]:
                attrs.pop(key, None)
        positional_args = []
        for key in self.positional_arguments():
            value = attrs.pop(key, None)
            positional_args.append(f'{value!r}')
        named_args = [f'{key}={getattr(self, key)!r}' for key in attrs]
        attr_str = ', '.join(positional_args + named_args)
        return f'{self.__class__.__name__}({attr_str})'

class System(PromptMessage):
    pass

class User(PromptMessage):
    pass

class Assistant(PromptMessage):
    pass


class FunctionCall(PromptMessage):
    def __init__(self, name: str, summarized_below=False, **args):
        # ATTENTION functions cannot have arguments with the same name as the
        # __init__ arguments.
        # That is 'name' and 'summarized_below'
        super().__init__('',
            role='assistant', name=name, summarized_below=summarized_below, args=args)

    def __repr__(self):
        args = self.args.copy()
        if self.summarized_below:
            args['summarized_below'] = True
        positional_args = [f"'{self.name}'"]
        named_args = [f'{key}={args[key]!r}' for key in args]
        attr_str = ', '.join(positional_args + named_args)
        return f'{self.__class__.__name__}({attr_str})'

    def plaintext(self) -> str:
        arguments = ", ".join(str(value) for value in self.args.values())
        return f"Action: {self.name}[{arguments}]"

    def openai_message(self) -> dict:
        arguments = dict(self.args)
        return {
            "role": "assistant",
            "content": '',
            "function_call": {
                "name": self.name,
                "arguments": json.dumps(arguments),
            },
        }

class FunctionResult(PromptMessage):

    def __init__(self, name: str, content: str, summarized_below=False):
        super().__init__(content, role='assistant', name=name, summarized_below=summarized_below)
    @classmethod
    def defaults(cls) -> dict:
        return {
            "role": 'assistant',
            "summarized_below": False,
        }

    @classmethod
    def positional_arguments(cls) -> List[str]:
        return ['name', 'content']

    def plaintext(self) -> str:
        return f"Observation: {self.get_content()}"

    def openai_message(self) -> dict:
        return {
            "role": "function",
            "name": self.name,
            "content": self.get_content(),
        }

class Prompt:
    def __init__(self, parts=None):
        if parts is None:
            parts = []
        self.parts = list(parts)

    def push(self, message: PromptMessage):
        self.parts.append(message)

    def to_messages(self) -> Any:
        raise NotImplementedError

    def __repr__(self):
        parts_repr = ',\n  '.join(repr(part) for part in self.parts)
        return f"{self.__class__.__name__}([\n  {parts_repr}\n])"
#    def __repr__(self):
#        parts_repr = pformat(self.parts, indent=2, width=80, depth=None)
#        return f"{self.__class__.__name__}({parts_repr})"

class FunctionalPrompt(Prompt):
    def to_messages(self) -> List[Dict[str, Any]]:
        return [part.openai_message() for part in self.parts]

    def function_call_from_response(self, response):
        if hasattr(response, 'tool_calls'):
            return response.tool_calls[0].function
        if hasattr(response, 'function_call'):
            return response.function_call
        return None


class PlainTextPrompt(Prompt):

    def to_text(self) -> str:
        return "\n".join(part.plaintext() for part in self.parts)

    def to_messages(self) -> List[Dict[str, Any]]:
        return [{ "role": "user", "content": self.to_text()}]


# Example Usage:

if __name__ == "__main__":

    # Using FunctionalPrompt
    fprompt = FunctionalPrompt([
        System("Solve a question answering task with interleaving Thought, Action and Observation steps."),
        User("\nQuestion: What is the terminal velocity of an unleaded swallow?"),
        FunctionCall('Search', reason="I need to search through my book of Monty Python jokes.", query="Monty Python"),
        FunctionResult('Search', "Here are all the Monty Python jokes you know: ..."),
        FunctionResult('Search', "Here are all the Monty Python jokes you know: ...", summarized_below=True),
        FunctionCall('Lookup', reason="Lookup the Swallow joke", keyword="Unleaded swallow"),
        User("Observation: Did you mean Unladen Swallow?"),
        FunctionCall('Finish', query='Oh you!'),
    ])
    print(pformat(fprompt.to_messages()))
    print(fprompt)
    #print(pformat(fprompt))
    exit()
    print()
    print("-" * 80)
    print()

    # Using PlainTextPrompt
    pprompt = PlainTextPrompt([
        System("Solve a question answering task with interleaving Thought, Action and Observation steps."),
        User("\nQuestion: What is the terminal velocity of an unleaded swallow?"),
        FunctionCall('Search', reason="I need to search through my book of Monty Python jokes.", query="Monty Python"),
        FunctionResult('Search', "Here are all the Monty Python jokes you know: ..."),
        FunctionResult('Search', reason="Here are all the Monty Python jokes you know: ...", summarized_below=True),
        Assistant("Lookup[Unleaded swallow]"),
        User("Observation: Did you mean Unladen Swallow?"),
        FunctionCall('Finish', query='Oh you!'),
    ])
    print(pprompt.to_text())
    #print(pformat(pprompt))


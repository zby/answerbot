import json
from typing import Any, Callable, Iterable, List, Dict
from pprint import pformat


class PromptMessage:
    def plaintext(self) -> str:
        raise Exception("plaintext() not implemented for", self.__class__.__name__)

    def openai_message(self) -> dict:
        raise Exception("openai_message() not implemented for", self.__class__.__name__)


class BasicPrompt(PromptMessage):
    def __init__(self, content: str):
        self.content = content

    def plaintext(self) -> str:
        return self.content

    def __repr__(self):
         class_name = self.__class__.__name__
         return f"{class_name}({repr(self.content)})"


class System(BasicPrompt):

    def openai_message(self) -> dict:
        return { "role": "system", "content": self.content }

class User(BasicPrompt):

    def openai_message(self) -> dict:
        return { "role": "user", "content": self.content }


class Assistant(BasicPrompt):

    def openai_message(self) -> dict:
        return { "role": "assistant", "content": self.content }

class FunctionCall(PromptMessage):
    def __init__(self, name: str, thought: str = None, **args):
        self.name = name
        self.thought = thought
        self.args = args

    def plaintext(self) -> str:
        arguments = ", ".join(self.args.values())
        thought = f"Thought: {self.thought}\n" if self.thought else ""
        return f"{thought}Action: {self.name}[{arguments}]"

    def openai_message(self) -> dict:
        arguments = dict(self.args)
        arguments["thought"] = self.thought
        return {
            "role": "assistant",
            "content": '',
            "function_call": {
                "name": self.name,
                "arguments": json.dumps(arguments),
            },
        }
    def __repr__(self):
        args_repr = ', '.join([f"{k}={repr(v)}" for k, v in self.args.items()])
        thought_repr = f"thought={self.thought!r}" if self.thought is not None else ''
        joined_repr = ', '.join(filter(None, [args_repr, thought_repr]))
        return f"{self.__class__.__name__}(name={self.name!r}, {joined_repr})"

class FunctionResult(PromptMessage):
    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content

    def plaintext(self) -> str:
        return f"Observation: {self.content}"

    def openai_message(self) -> dict:
        return {
            "role": "function",
            "name": self.name,
            "content": self.content,
        }

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, {pformat(self.content, width=120)})"

class Prompt:
    def __init__(self, parts: Iterable[PromptMessage] = []):
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

    def to_text(self):
        return pformat(self.to_messages())

    def function_call_from_response(self, response):
        return response.get("function_call")


class PlainTextPrompt(Prompt):


    def to_messages(self) -> List[Dict[str, Any]]:
        return [{ "role": "user", "content": self.to_text()}]


# Example Usage:

if __name__ == "__main__":

    # Using FunctionalPrompt
    fprompt = FunctionalPrompt([
        System("Solve a question answering task with interleaving Thought, Action and Observation steps."),
        User("\nQuestion: What is the terminal velocity of an unleaded swallow?"),
        FunctionCall('Search', thought="I need to search through my book of Monty Python jokes.", query="Monty Python"),
        FunctionResult('Search', "Here are all the Monty Python jokes you know: ..."),
        Assistant("Lookup[Unleaded swallow]"),
        User("Observation: Did you mean Unladen Swallow?"),
        FunctionCall('Finish', query='Oh you!'),
    ])
    print(fprompt.to_text())
    #print(pformat(fprompt))

    print()
    print("-" * 80)
    print()

    # Using PlainTextPrompt
    pprompt = PlainTextPrompt([
        System("Solve a question answering task with interleaving Thought, Action and Observation steps."),
        User("\nQuestion: What is the terminal velocity of an unleaded swallow?"),
        FunctionCall('Search', thought="I need to search through my book of Monty Python jokes.", query="Monty Python"),
        FunctionResult('Search', "Here are all the Monty Python jokes you know: ..."),
        Assistant("Lookup[Unleaded swallow]"),
        User("Observation: Did you mean Unladen Swallow?"),
        FunctionCall('Finish', query='Oh you!'),
    ])
    print(pprompt.to_text())
    #print(pformat(pprompt))


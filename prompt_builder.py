import json
from typing import Any, Callable, Iterable, List, Dict
from pprint import pformat


class PromptMessage:
    def plaintext(self) -> str:
        raise Exception("plaintext() not implemented for", self.__class__.__name__)

    def openai_message(self) -> dict:
        raise Exception("openai_message() not implemented for", self.__class__.__name__)

class System(PromptMessage):
    def __init__(self, content: str):
        self.content = content

    def plaintext(self) -> str:
        return self.content

    def openai_message(self) -> dict:
        return { "role": "system", "content": self.content }

class User(PromptMessage):
    def __init__(self, content: str):
        self.content = content

    def plaintext(self) -> str:
        return self.content

    def openai_message(self) -> dict:
        return { "role": "user", "content": self.content }


class Assistant(PromptMessage):
    def __init__(self, content: str):
        self.content = content

    def plaintext(self) -> str:
        return self.content

    def openai_message(self) -> dict:
        return { "role": "assistant", "content": self.content }

class FunctionCall(PromptMessage):
    def __init__(self, name: str, thought=None, **args):
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

class Prompt:
    def __init__(self, parts: Iterable[PromptMessage] = []):
        self.parts = list(parts)

    def push(self, message: PromptMessage):
        self.parts.append(message)

    def to_messages(self) -> Any:
        raise NotImplementedError

class FunctionalPrompt(Prompt):
    def to_messages(self) -> List[Dict[str, Any]]:
        return [part.openai_message() for part in self.parts]

    def to_text(self):
        return pformat(self.to_messages())

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
        FunctionCall('Search', thought="I need to search through my book of Monty Python jokes.", query="Monty Python"),
        FunctionResult('Search', "Here are all the Monty Python jokes you know: ..."),
        Assistant("Lookup[Unleaded swallow]"),
        User("Observation: Did you mean Unladen Swallow?"),
        FunctionCall('Finish', query='Oh you!'),
    ])
    print(fprompt.to_text())

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


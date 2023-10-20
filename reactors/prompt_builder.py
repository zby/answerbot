import json
from typing import Any, Callable, Iterable

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
        return f"{thought}{self.name}[{arguments}]"

    def openai_message(self) -> dict:
        arguments = dict(self.args)
        if self.thought is not None:
            arguments["thought"] = self.thought
        return {
            "role": "assistant",
            "content": self.plaintext(),
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

class OpenAIMessage(PromptMessage):
    def __init__(self, message: dict):
        self.message = message

    def plaintext(self) -> str:
        return self.message.get("content", "")

    def openai_message(self) -> dict:
        return self.message

Formatter = Callable[[Iterable[PromptMessage]], Any]

class OutputFormat:
    @staticmethod
    def plain(messages: Iterable[PromptMessage]) -> str:
        return "\n".join(map(lambda m: m.plaintext(), messages))

    @staticmethod
    def openai_messages(messages: Iterable[PromptMessage]) -> Iterable[dict]:
        """ For OpenAI's `messages` API, which expects a list of JSON objects, we return an iterable of dicts """
        return list(map(lambda m: m.openai_message(), messages))

class PromptBuilder:
    def __init__(self, parts: Iterable[PromptMessage]):
        self.parts = list(parts)

    def push(self, message: PromptMessage):
        self.parts.append(message)

    def build(self, method: Formatter) -> Any: # TODO return something more specific? Something openai_query can be sure to handle
        return method(self.parts)


if __name__ == "__main__":
    assert Assistant("Thought: I need to search through my book of Monty Python jokes.\nSearch[Monty Python]").plaintext() \
        == FunctionCall('Search', query="Monty Python", thought="I need to search through my book of Monty Python jokes.").plaintext()


    from pprint import pprint

    prompt = PromptBuilder([
        System("Solve a question answering task with interleaving Thought, Action and Observation steps."),
        User("Question: What is the terminal velocity of an unleaded swallow?"),
        FunctionCall('Search', query="Monty Python", thought="I need to search through my book of Monty Python jokes."),
        FunctionResult('Search', "Here are all the Monty Python jokes you know: ..."),
        Assistant("Lookup[Unleaded swallow]"),
        User("Observation: Did you mean Unladen Swallow?"),
        Assistant("Finish[Oh you!]"),
    ])

    pprint(prompt.build(OutputFormat.plain))
    # pprint(prompt.build(OutputFormat.openai_messages))

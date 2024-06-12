from openai.types.chat import ChatCompletionMessage

from functools import singledispatch

from typing import Union, Any
from pprint import pprint

class ToolResult:
    def to_message(self) -> dict:
        return {'node': 'this is a tool result'}

class TraceBase:
    pass

@singledispatch
def to_messages(obj) -> list[dict]:
    raise NotImplementedError(f'to_messages not implemented for type {type(obj)}')

@to_messages.register
def _(obj: dict) -> list[dict]:
    return [obj]

@to_messages.register
def _(obj: ToolResult) -> list[dict]:
    return [obj.to_message()]

@to_messages.register
def _(obj: ChatCompletionMessage) -> list[dict]:
    return [obj.to_dict()]

@to_messages.register
def _(obj: TraceBase) -> list[dict]:    
    return []

class Trace(TraceBase):
    def __init__(self, entries = None, question=None):
        self.entries: list[Union[dict, ChatCompletionMessage, ToolResult, Trace]] = [] if entries is None else entries
        self.question = question

    def append(self, entry):
        self.entries.append(entry)

    def add_message(self, role, content):
        self.entries.append({ 'role': role, 'content': content })

    def add_sub_trace(self, sub_trace, sub_trace_result):
        self.entries.append({'role': 'assistant', 'content': sub_trace_result})
        self.entries.append(sub_trace)

    def to_messages(self) -> list[dict]:
        """
        Returns:
        List[Dict]: A list of dictionaries representing the messages and tool results.
        """
        all_messages = [
            {'role': 'user', 'content': f"Question: {self.question}"}
        ]
        for entry in self.entries:
            all_messages.extend(to_messages(entry))
        return all_messages



# Example usage
if __name__ == "__main__":
    trace = Trace([], 'What is the distance from Moscow to Sao Paulo?')
    trace.append({'node1': 'value1'})
    trace.append(ToolResult())

    sub_trace = Trace([], trace.question)
    trace.add_sub_trace(sub_trace, 'sub_trace_result')

    # print main trace messages
    pprint(trace.to_messages())

    pprint(sub_trace.to_messages())

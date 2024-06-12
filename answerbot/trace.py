from openai.types.chat import ChatCompletionMessage

from typing import Union, Any
from pprint import pprint

class ToolResult:
    def to_message(self) -> dict:
        return {'node': 'this is a tool result'}

class Trace:
    def __init__(self, entries = None, question=None, result=None):
        self.entries: list[Union[dict, ChatCompletionMessage, ToolResult, Trace]] = [] if entries is None else entries
        self.question = question
        self.result = result

    def append(self, entry):
        self.entries.append(entry)

    def add_message(self, role, content):
        self.entries.append({ 'role': role, 'content': content })

    def add_sub_trace(self, sub_trace):
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
            if isinstance(entry, Trace):
                if entry.result is not None:
                    all_messages.append({'role': 'assistant', 'content': entry.result})
            elif isinstance(entry, ToolResult):
                all_messages.append(entry.to_message())
            elif isinstance(entry, ChatCompletionMessage):
                all_messages.append(entry.to_dict())
            elif isinstance(entry, dict):
                all_messages.append(entry)
            else:
                raise ValueError(f'Unsupported entry type: {type(entry)}')
        return all_messages



# Example usage
if __name__ == "__main__":
    trace = Trace([], 'What is the distance from Moscow to Sao Paulo?')
    trace.append({'node1': 'value1'})
    trace.append(ToolResult())

    sub_trace = Trace([], trace.question, 'sub_trace_result')
    trace.add_sub_trace(sub_trace)

    # print main trace messages
    pprint(trace.to_messages())

    pprint(sub_trace.to_messages())

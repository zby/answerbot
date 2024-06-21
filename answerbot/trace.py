from litellm.types.utils import Message
from dataclasses import dataclass, field

from typing import Union, Any, Optional
from pprint import pprint

from llm_easy_tools import ToolResult

@dataclass
class Question:
    question: str
    
    def to_message(self):
        return {'role': 'user', 'content': f"Question: {self.question}" }

@dataclass
class Trace:
    entries: list[Union[dict, Question, Message, ToolResult, 'Trace']] = field(default_factory=list)
    result: Optional[dict] = None

    def __init__(self, entries=None, result=None):
        self.entries: list[Union[dict, Question, Message, ToolResult, Trace]] = [] if entries is None else entries
        self.result: Optional[dict] = result

    def append(self, entry):
        self.entries.append(entry)

    def to_messages(self) -> list[dict]:
        """
        Returns:
        List[Dict]: A list of dictionaries representing the messages and tool results.
        """
        all_messages = []
        for entry in self.entries:
            if isinstance(entry, Trace):
                if entry.result is not None:
                    all_messages.append(entry.result)
            elif isinstance(entry, ToolResult):
                all_messages.append(entry.to_message())
            elif isinstance(entry, Message):
                all_messages.append(entry.model_dump())
            elif isinstance(entry, dict):
                all_messages.append(entry)
            elif isinstance(entry, Question):
                all_messages.append(entry.to_message())
            else:
                raise ValueError(f'Unsupported entry type: {type(entry)}')
        return all_messages

    def user_question(self):
        for entry in self.entries:
            if isinstance(entry, Question):
                return entry.question
        return None

    def __str__(self):
        entries_str = ''
        for entry in self.entries:
            entry_str = str(entry)
            entry_str = entry_str.replace('\n', '\n    ')
            entries_str += f"\n    {entry_str}"
        return f"Trace(\n    entries=[{entries_str}], \n    result={str(self.result)} \n)"

    def length(self):
        length = 0
        for entry in self.entries:
            if isinstance(entry, Trace):
                length += entry.length()
            else:
                length += 1
        return length



# Example usage
if __name__ == "__main__":
    trace = Trace([Question('What is the distance from Moscow to Sao Paulo?')])
    trace.append({'node1': 'value1'})
    trace.append(ToolResult(tool_call_id='tool_call_id', name='tool_name', output='tool_result'))

    sub_trace = Trace([trace.entries[0]], {'role': 'assistant', 'content': 'sub_trace_result'})
    trace.append(sub_trace)

    # print main trace messages
    pprint(trace.to_messages())

    pprint(sub_trace.to_messages())

    print(str(trace))

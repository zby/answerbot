from typing import Union, Any
from pydantic import BaseModel
from answerbot.tools.observation import Observation
from llm_easy_tools import ToolResult
from openai.types.chat import ChatCompletionMessage

from typing import Union, Any
from llm_easy_tools import ToolResult

class Node:
    def __init__(self, value: Union[dict, ToolResult]) -> None:
        self.value = value

    def to_message(self) -> dict:
        if isinstance(self.value, dict):
            return self.value
        elif isinstance(self.value, ToolResult):
            return self.value.to_message()

class Trace:
    def __init__(self, value: Union[None, dict] = None) -> None:
        self.value = value
        self.children: list[Union[Node, 'Trace']] = []
        self.parent: Union[None, 'Trace'] = None
        self.roots: list['Trace'] = [self]  # Track the latest root in this trace

    def _set_value(self, value: dict) -> None:
        self.value = value

    def to_message(self) -> dict:
        if self.value is None:
            raise ValueError("Trace value is None and the trace is not closed")
        return self.value

    def messages_trace(self) -> list[dict]:
        # Return the value of calling to_message on the children of the last element of the roots list
        return [child.to_message() for child in self._current_root().children]

    def attach(self, value: dict) -> None:
        # Create a new Node with the given value and attach it to the last element in the roots list
        new_node = Node(value)
        self._current_root().children.append(new_node)

    def attach_sub_trace(self) -> 'Trace':
        # Create a new sub_trace and attach it to the last element in the roots list
        sub_trace = Trace()
        sub_trace.parent = self._current_root()
        self._current_root().children.append(sub_trace)
        self.roots.append(sub_trace)
        return sub_trace

    def close(self, value: dict) -> None:
        self._current_root()._set_value(value)
        self.roots.pop()

    def _current_root(self) -> 'Trace':
        # Return the last element from the roots list
        return self.roots[-1]


# Example usage
if __name__ == "__main__":
    trace = Trace()
    trace.attach({"node1": "value1"})
    trace.attach({"node2": "value2"})

    print("Trace messages_trace before attaching sub_trace:", trace.messages_trace())

    sub_trace = trace.attach_sub_trace()
    trace.attach({"node3": "value3"})  # This should attach node3 to the sub_trace

    print("Trace messages_trace after attaching sub_trace and node3:", trace.messages_trace())

    trace.close({"sub_trace": "value"})

    print("Trace messages_trace after closing sub_trace:", trace.messages_trace())

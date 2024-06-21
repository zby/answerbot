import pytest
from answerbot.trace import Trace, Question
from answerbot.tools.wiki_tool import ToolResult
from litellm.types.utils import Message

def test_append():
    sample_trace = Trace()
    entry = {"role": "user", "content": "What is the capital of France?"}
    sample_trace.append(entry)
    assert sample_trace.entries[-1] == entry
def test_question():
    sample_trace = Trace([Question("What is the capital of France?")])
    assert sample_trace.to_messages()[0] == {"role": "user", "content": "Question: What is the capital of France?"}
    sample_trace.append({"role": "assistant", "content": "Paris"})
    assert sample_trace.user_question() == "What is the capital of France?"

def test_to_messages():
    sample_trace = Trace([Question("What is the capital of France?")])
    sample_trace.append(Trace([], result={"role": "assistant", "content": "Paris"}))
    messages = sample_trace.to_messages()
    assert messages == [
        {"role": "user", "content": "Question: What is the capital of France?"},
        {"role": "assistant", "content": "Paris"}
    ]

if __name__ == "__main__":
    pytest.main()

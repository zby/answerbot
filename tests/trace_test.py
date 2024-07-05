import pytest
from answerbot.trace import Trace, Question
from answerbot.tools.wiki_tool import Observation, InfoPiece
from answerbot.clean_reflection import ReflectionResult, KnowledgeBase

from llm_easy_tools.processor import ToolResult


user_prompt_template = "Question: {question}\nMax LLM calls: {max_llm_calls}"
user_question = Question(user_prompt_template, question='What is the capital of France?', max_llm_calls=3)

def test_append():
    trace = Trace()
    entry = {"role": "user", "content": "What is the capital of France?"}
    trace.append(entry)
    assert trace.entries[-1] == entry

def test_question():
    trace = Trace([user_question])
    assert trace.to_messages()[0] == {"role": "user", "content": "Question: What is the capital of France?\nMax LLM calls: 3"}
    trace.append({"role": "assistant", "content": "Paris"})
    assert trace.user_question() == "What is the capital of France?"

def test_to_messages():
    main_trace = Trace([user_question])

    # Test simple dictionary entry
    main_trace.append({"role": "assistant", "content": "The capital of France is Paris."})
    messages = main_trace.to_messages()
    assert len(messages) == 2
    assert messages[1] == {"role": "assistant", "content": "The capital of France is Paris."}

    sub_trace_question = Question(user_prompt_template, question="What is the population of Paris?", max_llm_calls=3)

    # Test sub_trace without answer
    sub_trace_no_answer = Trace([sub_trace_question])
    sub_trace_no_answer.append({"role": "assistant", "content": "I need to look that up."})
    main_trace.append(sub_trace_no_answer)
    messages = main_trace.to_messages()
    assert len(messages) == 2  # adding a sub trace without answer should not mean we add more messages

    # Test sub_trace with answer
    sub_trace_with_answer = Trace([sub_trace_question])
    sub_trace_with_answer.append({"role": "assistant", "content": "The population of Paris is approximately 2.2 million."})
    sub_trace_with_answer.finish("2.2 million", "Based on recent census data.")
    main_trace.append(sub_trace_with_answer)
    messages = main_trace.to_messages()
    assert len(messages) == 3
    report = messages[2]['content']
    assert "2.2 million" in report
    assert "Based on recent census data." in report

def test_sub_trace_wrapped_in_tool_result():
    main_trace = Trace([user_question])
    sub_trace_question = Question(user_prompt_template, question="What is the population of Paris?", max_llm_calls=3)
    sub_trace_with_answer = Trace([sub_trace_question])
    sub_trace_with_answer.append({"role": "assistant", "content": "The population of Paris is approximately 2.2 million."})
    sub_trace_with_answer.finish("2.2 million", "Based on recent census data.")
    tool_result = ToolResult(
        tool_call_id="123",
        name="population_lookup",
        output=sub_trace_with_answer
    )
    main_trace.append(tool_result)

    messages = main_trace.to_messages()
    assert len(messages) == 2
    report = messages[1]['content']
    assert "2.2 million" in report
    assert "Based on recent census data." in report


class SimpleClass:
    def __init__(self, field1, field2):
        self.field1 = field1
        self.field2 = field2

    def __str__(self):
        return f"SimpleClass(field1={self.field1}, field2={self.field2})"

def test_trace_to_messages_with_tool_result_simple_class():
    main_trace = Trace([user_question])
    simple_obj = SimpleClass("test_value", 123)
    tool_result = ToolResult(
        tool_call_id="456",
        name="simple_class_tool",
        output=simple_obj
    )
    main_trace.append(tool_result)

    messages = main_trace.to_messages()
    assert len(messages) == 2
    assert messages[0] == user_question.to_message()
    assert messages[1]['role'] == 'tool'
    assert messages[1]['content'] == str(simple_obj)
    assert messages[1]['tool_call_id'] == tool_result.tool_call_id
    assert messages[1]['name'] == tool_result.name


def test_trace_length():
    trace = Trace([user_question, {"role": "assistant", "content": "Paris"}])
    sub_trace = Trace([{"role": "system", "content": "Sub trace content"}])
    trace.append(sub_trace)
    assert trace.length() == 3  # 2 from main trace + 1 from sub trace


if __name__ == "__main__":
    pytest.main()

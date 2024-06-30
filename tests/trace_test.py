import pytest
from answerbot.trace import Trace, Question
from answerbot.tools.wiki_tool import Observation, InfoPiece
from answerbot.clean_reflection import ReflectionResult, KnowledgeBase

from llm_easy_tools.processor import ToolResult


user_prompt_template = "Question: {question}\nMax LLM calls: {max_llm_calls}"
user_question = Question(user_prompt_template, "What is the capital of France?", 3)

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

    sub_trace_question = Question(user_prompt_template, "What is the population of Paris?", 3)

    # Test sub_trace without answer
    sub_trace_no_answer = Trace([sub_trace_question])
    sub_trace_no_answer.append({"role": "assistant", "content": "I need to look that up."})
    main_trace.append(sub_trace_no_answer)
    messages = main_trace.to_messages()
    assert len(messages) == 2  # adding a sub trace without answer should not mean we add more messages

    # Test sub_trace with answer
    sub_trace_with_answer = Trace([sub_trace_question])
    sub_trace_with_answer.append({"role": "assistant", "content": "The population of Paris is approximately 2.2 million."})
    sub_trace_with_answer.set_answer("2.2 million", "2.2M", "Based on recent census data.")
    main_trace.append(sub_trace_with_answer)
    messages = main_trace.to_messages()
    assert len(messages) == 3
    report = messages[2]['content']
    assert "2.2 million" in report
    assert "Based on recent census data." in report

def test_sub_trace_wrapped_in_tool_result():
    main_trace = Trace([user_question])
    sub_trace_question = Question(user_prompt_template, "What is the population of Paris?", 3)
    sub_trace_with_answer = Trace([sub_trace_question])
    sub_trace_with_answer.append({"role": "assistant", "content": "The population of Paris is approximately 2.2 million."})
    sub_trace_with_answer.set_answer("2.2 million", "2.2M", "Based on recent census data.")
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

def test_trace_length():
    trace = Trace([user_question, {"role": "assistant", "content": "Paris"}])
    sub_trace = Trace([{"role": "system", "content": "Sub trace content"}])
    trace.append(sub_trace)
    assert trace.length() == 3  # 2 from main trace + 1 from sub trace

def test_update_knowledge_base():
    what_have_we_learned = KnowledgeBase()
    what_have_we_learned.add_info(
        url="https://something.com",
        quotes=["Something is something."],
        learned="Something is something."
    )
    trace = Trace(what_have_we_learned=what_have_we_learned)

    observation = Observation(
        info_pieces=[InfoPiece(text="Paris is the capital of France.", quotable=True)],
        current_url="https://example.com"
    )

    reflection_result = ReflectionResult(
        what_have_we_learned="Paris is the capital of France.",
        comment="We are learning something new.",
        new_sources=["https://newsource.com"],
        relevant_quotes=["Paris is the capital of France."]
    )
    knowledge_update_str = trace.update_knowledge_base(reflection_result, observation)
    assert "https://example.com" in knowledge_update_str
    print(trace.what_have_we_learned.urls())
    assert set(trace.what_have_we_learned.urls()) == {"https://something.com", "https://example.com"}


if __name__ == "__main__":
    pytest.main()

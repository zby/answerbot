import pytest
from answerbot.trace import Trace, Question
from answerbot.tools.wiki_tool import Observation, InfoPiece
from answerbot.clean_reflection import ReflectionResult, KnowledgeBase

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
def test_trace_length():
    trace = Trace([Question("What is the capital of France?"), {"role": "assistant", "content": "Paris"}])
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

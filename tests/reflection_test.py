import pytest
from answerbot.knowledgebase import KnowledgeBase
from answerbot.reflection_result import ReflectionResult

from answerbot.tools.observation import Observation

def test_stringification():
    reflection_result = ReflectionResult(
        what_have_we_learned="NEW INFORMATION",
        relevant_quotes=[
            "Fictional teenage girl Corliss Archer ..."
        ],
        new_sources=[
            "Janet Waldo",
            "Kiss and Tell (1945 film)"
        ],
        comment="The relevant quotes provide background ...",
    )

    # Stringify the object
    reflection_result_str = str(reflection_result)
    assert "Fictional teenage girl Corliss Archer" in reflection_result_str
    assert "Janet Waldo" in reflection_result_str
    assert "Kiss and Tell (1945 film)" in reflection_result_str
    assert "The relevant quotes provide background ..." in reflection_result_str
    #assert "NEW INFORMATION" in reflection_result_str

def test_empty_reflection_result_stringify():
    reflection_result_page = ReflectionResult(
        what_have_we_learned="",
        relevant_quotes=[],
        new_sources=[],
        comment="",
    )

    # Stringify the object
    reflection_result_page_str = str(reflection_result_page)
    assert reflection_result_page_str == ""

def test_remove_source_from_reflection_result():
    # Setup
    reflection_result = ReflectionResult(
        what_have_we_learned="NEW INFORMATION",
        relevant_quotes=["Quote about AI."],
        new_sources=["http://source1.com", "http://source2.com"],
        comment="Initial sources included."
    )

    # Action
    reflection_result.remove_checked_urls(["http://source1.com"])

    # Assert
    assert "http://source1.com" not in reflection_result.new_sources
    assert "http://source2.com" in reflection_result.new_sources
    assert len(reflection_result.new_sources) == 1

def test_check_base():
    # Setup
    observation = Observation(
        content="""Extended discussion on artificial *intelligence* and its impacts.
Brief mention of machine learning within broader tech trends.
Unrelated content about economics.
Unquotable.
Artificial intelligence could revolutionize many sectors.""",
        operation="test_operation",
        source="http://test.com",
        quotable=True
    )
    reflection_result = ReflectionResult(
        what_have_we_learned="Insights on AI",
        relevant_quotes=["artificial intelligence", "machine learning", "revolutionize"],
        new_sources=["http://source1.com"],
        comment="Reflection on AI"
    )

    # Action
    checked_knowledge_piece = reflection_result.extract_knowledge(observation)

    # Assert
    ai_quotes = [quote for quote in checked_knowledge_piece.quotes if "intelligence" in quote]
    assert len(ai_quotes) == 2, "Should have two quotes containing 'artificial intelligence': one from 'artificial *intelligence*' and one from 'Artificial intelligence could revolutionize many sectors.'"
    assert "machine learning" in ' '.join(checked_knowledge_piece.quotes), "Should find 'machine learning'"
    assert "revolutionize" in ' '.join(checked_knowledge_piece.quotes), "Should find 'revolutionize'"
    assert len(checked_knowledge_piece.quotes) == 4, "Should have four valid quotes checked"

def test_update_knowledge_base():
    what_have_we_learned = KnowledgeBase()
    what_have_we_learned.add_info(
        url="https://something.com",
        quotes=["Something is something."],
        learned="Something is something."
    )

    observation = Observation(
        content="Paris is the capital of France.",
        operation="test_operation",
        source="https://example.com",
        quotable=True
    )

    reflection_result = ReflectionResult(
        what_have_we_learned="Paris is the capital of France.",
        comment="We are learning something new.",
        new_sources=["https://newsource.com"],
        relevant_quotes=["Paris is the capital of France."]
    )
    knowledge_update_str = reflection_result.update_knowledge_base(what_have_we_learned, observation)
    assert "https://example.com" in knowledge_update_str
    print(what_have_we_learned.urls())
    assert set(what_have_we_learned.urls()) == {"https://something.com", "https://example.com"}
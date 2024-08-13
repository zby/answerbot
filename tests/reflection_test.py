import pytest
from answerbot.reflection_result import ReflectionResult, find_similar_fragments

from answerbot.tools.observation import Observation, KnowledgePiece, History

def test_remove_source_from_reflection_result():
    # Setup
    reflection_result = ReflectionResult(
        what_have_we_learned="NEW INFORMATION",
        relevant_quotes=["Quote about AI."],
        new_sources=["http://source1.com", "http://source2.com"],
        comment="Initial sources included."
    )

    # Action
    reflection_result.remove_checked_sources(["http://source1.com"])

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

def test_update_history():
    history = History()
    observation = Observation(
        content="Paris is the capital of France.",
        operation="test_operation",
        source="https://example.com",
        quotable=True
    )
    history.add_observation(observation)

    reflection_result = ReflectionResult(
        what_have_we_learned="Paris is the capital of France.",
        comment="We are learning something new.",
        new_sources=["https://newsource.com"],
        relevant_quotes=["Paris is the capital of France."]
    )
    reflection_result.update_history(history)
    assert history.knowledge_pieces[0].source == observation

def test_find_similar_fragments():
    text = "Maine, U.S."
    assert find_similar_fragments(text, 'U.S.') == ['U.S.']
    text = "[Capacity](https://en.wikipedia.org/wiki/Seating_capacity)| 3,677 (2,634 hockey)  \n> Construction  \n"
    quote = "Capacity: 3,677 (2,634 hockey)"
    assert find_similar_fragments(text, quote) == ['Capacity | 3,677 (2,634 hockey)']
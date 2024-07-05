import pytest
from answerbot.clean_reflection import ReflectionResult, KnowledgeBase
from answerbot.tools.wiki_tool import Observation, InfoPiece

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

#def test_reflection_result_default_new_sources():
#    # Create an instance without providing new_sources
#    reflection_result = ReflectionResult(
#        what_have_we_learned="Learned something important.",
#        comment="This is a test comment.",
#        relevant_quotes=["Quote1", "Quote2"]
#    )
#    
#    # Check if new_sources is an empty list by default
#    assert reflection_result.new_sources == [], "new_sources should default to an empty list if not provided"
def test_check_base():
    # Setup
    observation = Observation(
        info_pieces=[
            InfoPiece(text="Extended discussion on artificial *intelligence* and its impacts.", quotable=True, source="something"),
            InfoPiece(text="Brief mention of machine learning within broader tech trends.", quotable=True, source="something"),
            InfoPiece(text="Unrelated content about economics.", quotable=True, source="something"),
            InfoPiece(text="Unquotable.", quotable=False, source="something"),
            InfoPiece(text="Artificial intelligence could revolutionize many sectors.", quotable=True, source="something")
        ],
        current_url="http://test.com"
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
    assert checked_knowledge_piece.quotes.count("artificial intelligence") == 2, "Should match 'artificial intelligence' twice"
    assert "machine learning" in checked_knowledge_piece.quotes, "Should find 'machine learning'"
    assert "revolutionize" in checked_knowledge_piece.quotes, "Should find 'revolutionize'"
    assert len(checked_knowledge_piece.quotes) == 4, "Should have four valid quotes checked"

def test_update_knowledge_base():
    what_have_we_learned = KnowledgeBase()
    what_have_we_learned.add_info(
        url="https://something.com",
        quotes=["Something is something."],
        learned="Something is something."
    )

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
    knowledge_update_str = what_have_we_learned.update_knowledge_base(reflection_result, observation)
    assert "https://example.com" in knowledge_update_str
    print(what_have_we_learned.urls())
    assert set(what_have_we_learned.urls()) == {"https://something.com", "https://example.com"}


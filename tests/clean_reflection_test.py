import pytest
from answerbot.clean_reflection import ReflectionResult 
from answerbot.wiki_tool import Observation, InfoPiece

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
    reflection_result.remove_source("http://source1.com")

    # Assert
    assert "http://source1.com" not in reflection_result.new_sources
    assert "http://source2.com" in reflection_result.new_sources
    assert len(reflection_result.new_sources) == 1

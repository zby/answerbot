import pytest
from answerbot.clean_reflection import ReflectionResult 
from answerbot.wiki_tool import Observation, InfoPiece

def test_reflection_result_refine_observation():
    # Setup
    original_info_pieces = [
        InfoPiece(text="This is a relevant quote from the source. With additional text.", quotable=True),
        InfoPiece(text="This is another piece of information.", quotable=True),
        InfoPiece(text="This is a third piece of information.", quotable=False)
    ]
    observation = Observation(info_pieces=original_info_pieces)
    reflection_result = ReflectionResult(
        relevant_quotes=["This is a relevant quote from the source."],
        new_sources=["http://newsource.com"],
        comment="Check this new source."
    )

    # Action
    refined_observation = reflection_result.refine_observation(observation)

    # Assert
    assert len(refined_observation.info_pieces) == 2
    assert refined_observation.info_pieces[0].text == "This is a relevant quote from the source."
    assert refined_observation.info_pieces[1].text == "This is a third piece of information."
    assert refined_observation.interesting_links == ["http://newsource.com"]
    assert refined_observation.comment == "Check this new source."
    assert refined_observation.is_refined

def test_stringification():
    reflection_result = ReflectionResult(
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

def test_empty_reflection_result_stringify():
    reflection_result_page = ReflectionResult(
        relevant_quotes=[],
        new_sources=[],
        comment="",
    )

    # Stringify the object
    reflection_result_page_str = str(reflection_result_page)
    assert reflection_result_page_str == ""

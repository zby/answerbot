import pytest
import json
from unittest.mock import MagicMock, Mock
from answerbot.get_wikipedia import ContentRecord, Document
from answerbot.toolbox import WikipediaSearch, ToolBox, ToolResult

# Mock function to simulate a tool
def mock_tool_function(tool_args, **kwargs):
    return {"mocked_data": "test"}

def create_mock_function_call(name, arguments):
    function_call = type('MockFunctionCall', (object,), {})()  # Creating a mock function call object
    function_call.name = name
    function_call.arguments = json.dumps(arguments)
    return function_call


# Test for the 'finish' functionality
def test_finish_functionality():
    toolbox = ToolBox()
    function_call = create_mock_function_call("finish", {"answer": "Yes"})
    result = toolbox.process(function_call)
    assert result.tool_name == "finish"
    assert result.observations == "yes"

# Test for unknown tool name
def test_unknown_tool_name():
    toolbox = ToolBox()
    function_call = create_mock_function_call("unknown_tool", {})
    result = toolbox.process(function_call)
    assert result.tool_name == "unknown_tool"
    assert result.error == f"Unknown tool name: unknown_tool"

# Test a known tool name
def test_known_tool_name():
    toolbox = ToolBox()
    toolbox.function_mapping["mock_tool"] = mock_tool_function
    function_call = create_mock_function_call("mock_tool", {})
    result = toolbox.process(function_call)
    assert result.tool_name == "mock_tool"
    assert "mocked_data" in result.observations
    assert result.observations["mocked_data"] == "test"

@pytest.fixture
def wiki_search():
    wiki_api = MagicMock()
    return WikipediaSearch(wiki_api)


def test_lookup_with_no_document(wiki_search):
    function_args = {"keyword": "Python", "reason": "Test lookup"}
    test_response = wiki_search.lookup(function_args)
    assert test_response == "No document defined, cannot lookup"

def test_lookup_with_document(wiki_search):
    mock_document = MagicMock(spec=Document)
    mock_document.lookup_results = ["Mock text"]
    mock_document.lookup.return_value = "Mock text"
    wiki_search.search({'query': 'Python', 'reason': 'Test search'})
    wiki_search.document = mock_document
    function_args = {'keyword': 'Python', 'reason': 'Test lookup'}
    test_response = wiki_search.lookup(function_args)
    assert test_response == 'Keyword "Python" found on current page in 1 places. The first occurence:\nMock text'


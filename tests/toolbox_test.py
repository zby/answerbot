import pytest
import json
from unittest.mock import MagicMock, Mock
from answerbot.get_wikipedia import ContentRecord, Document
from answerbot.toolbox import WikipediaSearch, ToolBox, ToolResult

@pytest.fixture
def api_instance():
    return ToolBox()



# Mock function to simulate a tool
def mock_tool_function(**tool_args):
    return {"mocked_data": "test"}

def create_mock_function_call(name, arguments):
    function_call = type('MockFunctionCall', (object,), {})()  # Creating a mock function call object
    function_call.name = name
    function_call.arguments = json.dumps(arguments)
    return function_call


# Test for the 'finish' functionality
def test_finish_functionality():
    toolbox = ToolBox()
    function_call = create_mock_function_call("finish", {"answer": "Yes", "reason": "Why"})
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
    class MyToolBox(ToolBox):
        def mock_tool(self):
            """
            Mock tool description
            """
            return {"mocked_data": "test"}
    toolbox = MyToolBox()
    function_call = create_mock_function_call("mock_tool", {})
    result = toolbox.process(function_call)
    assert result.tool_name == "mock_tool"
    assert "mocked_data" in result.observations
    assert result.observations["mocked_data"] == "test"


def test_subclass():
    wiki_api = MagicMock()
    ws = WikipediaSearch(wiki_api)
    assert isinstance(ws, WikipediaSearch)

@pytest.fixture
def wiki_search():
    wiki_api = MagicMock()
    return WikipediaSearch(wiki_api)


def test_lookup_with_no_document(wiki_search):
    param = WikipediaSearch.Search(query="Python", reason="Test lookup")
    test_response = wiki_search.lookup(param)
    assert test_response == "No document defined, cannot lookup"

def test_lookup_with_document(wiki_search):
    mock_document = MagicMock(spec=Document)
    mock_document.lookup_results = ["Mock text"]
    mock_document.lookup.return_value = "Mock text"
    param = WikipediaSearch.Search(query='Python', reason='Test search')
    wiki_search.search(param)
    wiki_search.document = mock_document
    param = WikipediaSearch.Lookup(keyword='Python', reason='Test lookup')
    test_response = wiki_search.lookup(param)
    assert test_response == 'Keyword "Python" found on current page in 1 places. The first occurence:\nMock text'

def test_functions(wiki_search):
    function_dict = {}
    for tool in wiki_search.tools:
        function = tool["function"]
        assert function["name"] in ["search", "get", "lookup", "next", "finish"]
        function_dict[function["name"]] = function

    assert function_dict["finish"]["description"] == "Finish the task and return the answer."
    assert function_dict["finish"]["parameters"]["type"] == "object"
    assert function_dict["finish"]["parameters"]["properties"]["answer"]["type"] == "string"
    assert function_dict["finish"]["parameters"]["properties"]["answer"]["description"] == "The answer to the user's question"
    assert function_dict["finish"]["parameters"]["properties"]["reason"]["type"] == "string"
    assert function_dict["finish"]["parameters"]["properties"]["reason"]["description"] == "The reasoning behind the answer"

    assert function_dict["search"]["description"] == "Searches Wikipedia, saves the first result page, and informs about the content of that page."
    assert function_dict["search"]["parameters"]["type"] == "object"
    assert function_dict["search"]["parameters"]["properties"]["query"]["type"] == "string"
    assert function_dict["search"]["parameters"]["properties"]["query"]["description"] == "The query to search for on Wikipedia"
    assert function_dict["search"]["parameters"]["properties"]["reason"]["type"] == "string"
    assert function_dict["search"]["parameters"]["properties"]["reason"]["description"] == "The reason for searching"

    assert function_dict["search"]["parameters"]["required"] == ["reason", "query"]


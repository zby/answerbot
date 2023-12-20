import pytest
import json
from unittest.mock import MagicMock, Mock
from answerbot.get_wikipedia import ContentRecord, Document
from answerbot.toolbox import WikipediaSearch, ToolBox, ToolResult

@pytest.fixture
def api_instance():
    return ToolBox()


def test_eval_format(api_instance):
    docstring = """
    "name": "search",
    "description": "This function does something.",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "The first parameter.",
            },
            "param2": {
                "type": "string",
                "description": "The second parameter.",
            }
        },
        "required": ["param1", "param2"],
    },
    
    """
    function_info = api_instance._parse_docstring('search', docstring, 'eval')
    description = function_info['description']
    params = function_info['parameters']['properties']
    required = function_info['parameters']['required']
    assert description == "This function does something."
    assert 'param1' in params
    assert params['param1']['description'] == "The first parameter."
    assert params['param1']['type'] == "string"
    assert 'param2' in params
    assert params['param2']['description'] == "The second parameter."
    assert params['param2']['type'] == "string"
    assert 'param1' in required
    assert 'param2' in required


def test_rest_format(api_instance):
    docstring = """
    This function does something.
    
    Args:
        param1 (str): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.
    """
    function_info = api_instance._parse_docstring('search', docstring, 'rest')
    description = function_info['description']
    params = function_info['parameters']['properties']
    required = function_info['parameters']['required']
    assert description == "This function does something."
    assert 'param1' in params
    assert params['param1']['description'] == "The first parameter."
    assert params['param1']['type'] == "string"
    assert 'param2' in params
    assert params['param2']['description'] == "The second parameter."
    assert params['param2']['type'] == "string"
    assert 'param1' in required
    assert 'param2' in required

def test_google_format(api_instance):
    docstring = """
    This function does something.

    Args:
        param1 (str): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.
    """
    function_info = api_instance._parse_docstring('search', docstring, 'google')
    description = function_info['description']
    params = function_info['parameters']['properties']
    required = function_info['parameters']['required']
    assert description == "This function does something."
    assert 'param1' in params
    assert params['param1']['description'] == "The first parameter."
    assert params['param1']['type'] == "string"
    assert 'param2' in params
    assert params['param2']['description'] == "The second parameter."
    assert params['param2']['type'] == "string"
    assert 'param1' in required
    assert 'param2' in required

#def test_numpy_format(api_instance):
#    docstring = """
#    This function does something.
#
#    Parameters
#    ----------
#    param1 : int
#        The first parameter.
#    param2 : str
#        The second parameter.
#
#    Returns
#    -------
#    bool
#        The return value. True for success, False otherwise.
#    """
#    description, params, required = api_instance._parse_docstring(docstring, 'numpy')
#    assert description == "This function does something."
#    assert 'param1' in params
#    assert params['param1']['description'] == "The first parameter."
#    assert 'param2' in params
#    assert params['param2']['description'] == "The second parameter."
#    assert 'param1' in required
#    assert 'param2' in required


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
    test_response = wiki_search.lookup(**function_args)
    assert test_response == "No document defined, cannot lookup"

def test_lookup_with_document(wiki_search):
    mock_document = MagicMock(spec=Document)
    mock_document.lookup_results = ["Mock text"]
    mock_document.lookup.return_value = "Mock text"
    wiki_search.search(query='Python', reason='Test search')
    wiki_search.document = mock_document
    function_args = {'keyword': 'Python', 'reason': 'Test lookup'}
    test_response = wiki_search.lookup(**function_args)
    assert test_response == 'Keyword "Python" found on current page in 1 places. The first occurence:\nMock text'

def test_functions(wiki_search):
    function_dict = {}
    for function in wiki_search.functions:
        assert function["name"] in ["search", "get", "lookup", "next", "finish"]
        function_dict[function["name"]] = function

    assert function_dict["finish"]["description"] == "Finish the task and return the answer."
    assert function_dict["finish"]["parameters"]["type"] == "object"
    assert function_dict["finish"]["parameters"]["properties"]["answer"]["type"] == "string"
    assert function_dict["finish"]["parameters"]["properties"]["answer"]["description"] == "The answer to the user's question."
    assert function_dict["finish"]["parameters"]["properties"]["reason"]["type"] == "string"
    assert function_dict["finish"]["parameters"]["properties"]["reason"]["description"] == "The reasoning behind the answer."

    assert function_dict["search"]["description"] == "Searches Wikipedia, saves the first result page, and informs about the content of that page."
    assert function_dict["search"]["parameters"]["type"] == "object"
    assert function_dict["search"]["parameters"]["properties"]["query"]["type"] == "string"
    assert function_dict["search"]["parameters"]["properties"]["query"]["description"] == "The query to search for on Wikipedia."
    assert function_dict["search"]["parameters"]["properties"]["reason"]["type"] == "string"
    assert function_dict["search"]["parameters"]["properties"]["reason"]["description"] == "The reason for searching."

    assert function_dict["search"]["parameters"]["required"] == ["query", "reason"]


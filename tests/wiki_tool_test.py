import pytest

from answerbot.tools.wiki_tool import WikipediaTool, Observation, BASE_URL
from answerbot.tools.markdown_document import MarkdownDocument
from unittest.mock import MagicMock, patch

from pprint import pprint

class MockHttpResponse:
    def __init__(self, text, status_code):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("HTTP Error")


# Test successful page retrieval
@patch('answerbot.tools.wiki_tool.requests.get')
def test_get_page_success(mock_get):
    mock_get.return_value = MockHttpResponse('<html><div id="bodyContent">Page Content</div></html>', 200)
    wiki_search = WikipediaTool()
    observation = wiki_search.get_url(BASE_URL)
    assert wiki_search.document is not None
    assert f"Successfully retrieved document from url: '{BASE_URL}'" in observation.content
    assert "The retrieved page starts with:" in observation.content
    assert "> Page Content" in observation.content
    assert "Page Content" in wiki_search.document.text


@patch('answerbot.tools.wiki_tool.requests.get')
def test_get_page_404(mock_get):
    mock_get.return_value = MockHttpResponse("Page not found", 404)
    wiki_search = WikipediaTool()
    observation = wiki_search.get_url(BASE_URL)
    assert wiki_search.document is None
    assert "Page does not exist." in observation.content

# Test other HTTP errors
@patch('answerbot.tools.wiki_tool.requests.get')
def test_get_page_http_error(mock_get):
    mock_get.return_value = MockHttpResponse("Error", 500)
    wiki_search = WikipediaTool()
    observation = wiki_search.get_url(BASE_URL)
    assert wiki_search.document is None
    assert "HTTP error occurred: 500" in observation.content
    assert "Retries exhausted. No options available." in observation.content

# Test successful search
@patch('answerbot.tools.wiki_tool.requests.get')
def test_search_success(mock_get):
    mock_http_response = MagicMock()
    mock_http_response.json.return_value = {
        'query': {
            'search': [{'title': 'TestTitle1'}, {'title': 'TestTitle2'}]
        }
    }
    mock_http_response.raise_for_status.return_value = None
    mock_get.return_value = mock_http_response

    wiki_tool = WikipediaTool()
    observation = wiki_tool.search("test_query")

    # Check if the search results text is correct
    assert "Wikipedia search results for query: 'test_query' are:" in observation.content
    assert "- [TestTitle1](https://en.wikipedia.org/wiki/TestTitle1)" in observation.content
    assert "- [TestTitle2](https://en.wikipedia.org/wiki/TestTitle2)" in observation.content

def test_observation_stringification():
    # Create an Observation
    observation = Observation(
        content="First piece of information\nSecond piece of information",
        source="Test URL",
        operation="search",
        quotable=True
    )

    # Convert the Observation to string and verify
    observation_str = str(observation)
    expected_str = """**Operation:** search


First piece of information
Second piece of information"""
    assert observation_str == expected_str, "The Observation stringification did not match the expected format."


def test_lookup_method():
    # Create a special document object with a controlled lookup method
    class MockDocument:
        def __init__(self, lookup_results):
            self.lookup_results = lookup_results
            self.lookup_word = None
            self.lookup_position = None

        def lookup(self, keyword):
            self.lookup_word = keyword
            self.lookup_position = 0
            if len(self.lookup_results) > 0:
                return self.lookup_results[self.lookup_position]
            else:
                return None


    # one lookup result
    mock_document = MockDocument(["This is a test lookup result for the keyword."])
    wiki_tool = WikipediaTool(document=mock_document, current_url="https://en.wikipedia.org")

    result = wiki_tool.lookup("test_keyword")

    assert 'Keyword "test_keyword" found' in result.content
    assert 'in 1 places' in result.content
    assert 'https://en.wikipedia.org' in result.content
    assert '> This is a test lookup result for the keyword.' in result.content
    assert '- *1 of 1 places*' in result.content
    assert "If you want to continue reading from this point, you can call `read_more`." in result.content

    # two lookup results
    mock_document = MockDocument(["This is a test lookup result for the keyword.", "This is a test lookup result for the keyword."])
    wiki_tool = WikipediaTool(document=mock_document)

    result = wiki_tool.lookup("test_keyword")

    assert 'Keyword "test_keyword" found' in result.content
    assert 'in 2 places' in result.content
    assert '> This is a test lookup result for the keyword.' in result.content
    assert '- *1 of 2 places*' in result.content
    assert "If you want to jump to the next occurence of the keyword you can call `next`." in result.content
    assert "If you want to continue reading from this point, you can call `read_more`." in result.content

    # no lookup results
    mock_document = MockDocument([])
    wiki_tool = WikipediaTool(document=mock_document)

    result = wiki_tool.lookup("test_keyword")

    assert f'Keyword "test_keyword" not found at' in result.content
    result = wiki_tool.lookup("test keyword")
    assert f'Keyword "test keyword" not found at' in result.content
    assert 'Note: Your keyword contains spaces.' in result.content
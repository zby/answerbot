import pytest

from answerbot.tools.wiki_tool import WikipediaTool, Observation, InfoPiece
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
@patch('answerbot.wiki_tool.requests.get')
def test_get_page_success(mock_get):
    mock_get.return_value = MockHttpResponse('<html><div id="bodyContent">Page Content</div></html>', 200)
    wiki_search = WikipediaTool()
    observation = wiki_search.get_url("https://www.test.test")
    assert wiki_search.document is not None
    assert observation.info_pieces[1].text == "Successfully retrieved document from url: 'https://www.test.test'"
    assert observation.info_pieces[2].text == "The retrieved page starts with:"
    assert observation.info_pieces[3].text == "> Page Content"
    assert "Page Content" in wiki_search.document.text


@patch('answerbot.wiki_tool.requests.get')
def test_get_page_404(mock_get):
    mock_get.return_value = MockHttpResponse("Page not found", 404)
    wiki_search = WikipediaTool()
    observation = wiki_search.get_url("https://www.test.test")
    assert wiki_search.document is None
    assert observation.info_pieces[1].text == "Page does not exist."

# Test other HTTP errors
@patch('answerbot.wiki_tool.requests.get')
def test_get_page_http_error(mock_get):
    mock_get.return_value = MockHttpResponse("Error", 500)
    wiki_search = WikipediaTool()
    observation = wiki_search.get_url("https://www.test.test")
    pprint(observation.info_pieces)
    assert wiki_search.document is None
    assert observation.info_pieces[2].text == "HTTP error occurred: 500"
    assert observation.info_pieces[3].text == "HTTP error occurred: 500"
    assert observation.info_pieces[wiki_search.max_retries + 1].text == "Retries exhausted. No options available."

# Test successful search
@patch('answerbot.wiki_tool.requests.get')
@patch('answerbot.wiki_tool.WikipediaTool.get_page')
def test_search_success(mock_get_page, mock_get):
    mock_http_response = MagicMock()
    mock_http_response.json.return_value = {
        'query': {
            'search': [{'title': 'TestTitle1'}, {'title': 'TestTitle2'}]
        }
    }
    mock_http_response.raise_for_status.return_value = None
    mock_get.return_value = mock_http_response

    # Mock get_page to return a dummy observation
    mock_get_page.return_value = Observation([
        InfoPiece(text="mock get page infopiece 1", source="https://en.wikipedia.org/wiki/TestTitle1"),
        InfoPiece(text="mock get page infopiece 2", source="https://en.wikipedia.org/wiki/TestTitle1")
        ],
    )

    wiki_tool = WikipediaTool()
    observation = wiki_tool.search("test_query")

    # Check if the search results text is correct
    assert observation.info_pieces[1].text == """Wikipedia search results for query: 'test_query' are:
- [TestTitle1](https://en.wikipedia.org/wiki/TestTitle1)
- [TestTitle2](https://en.wikipedia.org/wiki/TestTitle2)"""
    mock_get_page.assert_called_once_with('TestTitle1')
    # Check if the content from get_page is included in the observation
    assert observation.info_pieces[2].text == "mock get page infopiece 1"
    assert observation.info_pieces[3].text == "mock get page infopiece 2"


def test_observation_stringification():

    # Create an Observation with multiple InfoPieces
    info_pieces = [
        InfoPiece(text="First piece of information\nSecond piece of information", source="First URL"),
        InfoPiece(text="Third piece of information\nFourth piece of information", source="Second URL")
    ]
    observation = Observation(info_pieces=info_pieces)

    # Convert the Observation to string and verify
    observation_str = str(observation)
    expected_str = """

First piece of information
Second piece of information
— *from First URL*

Third piece of information
Fourth piece of information
— *from Second URL*"""
    assert observation_str == expected_str, "The Observation stringification did not match the expected format."
    print(observation_str)


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

    assert 'Keyword "test_keyword" found' in result.info_pieces[0].text
    assert 'in 1 places' in result.info_pieces[0].text
    assert 'https://en.wikipedia.org' in result.info_pieces[0].text
    assert result.info_pieces[1].text == "> This is a test lookup result for the keyword."
    assert result.info_pieces[2].text == "- *1 of 1 places*"
    assert result.info_pieces[3].text == "If you want to continue reading from this point, you can call `read_more`."

    # two lookup results
    mock_document = MockDocument(["This is a test lookup result for the keyword.", "This is a test lookup result for the keyword."])
    wiki_tool = WikipediaTool(document=mock_document)

    result = wiki_tool.lookup("test_keyword")

    assert 'Keyword "test_keyword" found' in result.info_pieces[0].text
    assert 'in 2 places' in result.info_pieces[0].text
    assert result.info_pieces[1].text == "> This is a test lookup result for the keyword."
    assert result.info_pieces[2].text == "- *1 of 2 places*"
    assert "If you want to jump to the next occurence of the keyword you can call `next`." in result.info_pieces[3].text
    assert "If you want to continue reading from this point, you can call `read_more`." in result.info_pieces[3].text

    # no lookup results
    mock_document = MockDocument([])
    wiki_tool = WikipediaTool(document=mock_document)

    result = wiki_tool.lookup("test_keyword")

    assert f'Keyword "test_keyword" not found at' in result.info_pieces[0].text
    result = wiki_tool.lookup("test keyword")
    assert f'Keyword "test keyword" not found at' in result.info_pieces[0].text
    assert 'Note: Your keyword contains spaces.' in result.info_pieces[0].text

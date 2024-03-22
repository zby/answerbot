import pytest

from answerbot.wikipedia_tool import WikipediaSearch, ContentRecord
from answerbot.document import MarkdownDocument
from unittest.mock import MagicMock, patch

class MockHttpResponse:
    def __init__(self, text, status_code):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("HTTP Error")



@patch('answerbot.wikipedia_tool.requests.get')
def test_follow_link(mock_get):
    content = """# Some Title

A link to [test](https://www.test.test), and here is another link to [Google](https://www.google.com).
Don't forget to check [Different text OpenAI](https://www.openai.com) for more information.
"""
    doc = MarkdownDocument(content)
    wiki_search = WikipediaSearch(document=doc)

    new_document = MarkdownDocument("New Page")
    content_record = ContentRecord(new_document, ['Page retrieval history'])
    wiki_search.get_page = MagicMock(return_value=content_record)
    follow_object = WikipediaSearch.FollowLink(link='1', reason="because")

    mock_get.return_value = MockHttpResponse('<html><div id="bodyContent">Page Content</div></html>', 200)
    wiki_search.follow_link(follow_object)
    new_content = wiki_search.document.content
    assert new_content.startswith("Page Content")

# Test successful page retrieval
@patch('answerbot.wikipedia_tool.requests.get')
def test_get_page_success(mock_get):
    mock_get.return_value = MockHttpResponse('<html><div id="bodyContent">Page Content</div></html>', 200)
    wiki_search = WikipediaSearch()
    result = wiki_search.get_page("TestPage")
    assert result.document is not None
    assert "Page Content" in result.document.content
    assert "Successfully retrieved 'TestPage'" in result.retrieval_history[0]

@patch('answerbot.wikipedia_tool.requests.get')
def test_get_page_404(mock_get):
    mock_get.return_value = MockHttpResponse("Page not found", 404)
    wiki_search = WikipediaSearch()
    result = wiki_search.get_page("NonExistentPage")
    assert result.document is None
    assert "Page 'NonExistentPage' does not exist." in result.retrieval_history

# Test other HTTP errors
@patch('answerbot.wikipedia_tool.requests.get')
def test_get_page_http_error(mock_get):
    mock_get.return_value = MockHttpResponse("Error", 500)
    wiki_search = WikipediaSearch()
    result = wiki_search.get_page("ErrorPage")
    assert result.document is None
    assert "HTTP error occurred: 500" in result.retrieval_history

# Test retries exhausted
@patch('answerbot.wikipedia_tool.requests.get')
def test_get_page_retries_exhausted(mock_get):
    mock_get.side_effect = [MockHttpResponse("Error", 500) for _ in range(5)]  # Assuming MAX_RETRIES is 5
    wiki_search = WikipediaSearch()
    result = wiki_search.get_page("ErrorPage")
    assert result.document is None
    assert "Retries exhausted. No options available." in result.retrieval_history

# Mocking the response for a successful search
def mock_successful_search(*args, **kwargs):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'query': {
            'search': [{'title': 'TestTitle1'}, {'title': 'TestTitle2'}]
        }
    }
    mock_response.raise_for_status.return_value = None
    return mock_response

# Test successful search
@patch('answerbot.wikipedia_tool.requests.get')
@patch('answerbot.wikipedia_tool.WikipediaSearch.get_page')
def test_search_success(mock_get_page, mock_get):
    mock_http_response = MagicMock()
    mock_http_response.json.return_value = {
        'query': {
            'search': [{'title': 'TestTitle1'}, {'title': 'TestTitle2'}]
        }
    }
    mock_http_response.raise_for_status.return_value = None
    mock_get.return_value = mock_http_response
    mock_get_page.return_value = ContentRecord(None, ['Page retrieval history'])
    wiki_search = WikipediaSearch()
    result = wiki_search.wiki_api_search("test_query")
    assert "Wikipedia search results for query: 'test_query'" in result.retrieval_history[0]
    assert mock_get_page.called


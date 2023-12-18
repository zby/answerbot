import pytest
from unittest.mock import patch, MagicMock
from answerbot.get_wikipedia import WikipediaApi, WikipediaDocument, ContentRecord

SMALL_CHUNK_SIZE = 72


# Mocking the requests response
class MockResponse:
    def __init__(self, text, status_code):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("HTTP Error")

# Test successful page retrieval
@patch('answerbot.get_wikipedia.requests.get')
def test_get_page_success(mock_get):
    mock_get.return_value = MockResponse('<html><div id="bodyContent">Page Content</div></html>', 200)
    api = WikipediaApi()
    result = api.get_page("TestPage")
    assert result.document is not None
    assert "Successfully retrieved 'TestPage'" in result.retrieval_history[0]

# Test 404 error
@patch('answerbot.get_wikipedia.requests.get')
def test_get_page_404(mock_get):
    mock_get.return_value = MockResponse("Page not found", 404)
    api = WikipediaApi()
    result = api.get_page("NonExistentPage")
    assert result.document is None
    assert "Page 'NonExistentPage' does not exist." in result.retrieval_history

# Test other HTTP errors
@patch('answerbot.get_wikipedia.requests.get')
def test_get_page_http_error(mock_get):
    mock_get.return_value = MockResponse("Error", 500)
    api = WikipediaApi()
    result = api.get_page("ErrorPage")
    assert result.document is None
    assert "HTTP error occurred: 500" in result.retrieval_history

# Test retries exhausted
@patch('answerbot.get_wikipedia.requests.get')
def test_get_page_retries_exhausted(mock_get):
    mock_get.side_effect = [MockResponse("Error", 500) for _ in range(5)]  # Assuming MAX_RETRIES is 5
    api = WikipediaApi()
    result = api.get_page("ErrorPage")
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
@patch('answerbot.get_wikipedia.requests.get', side_effect=mock_successful_search)
@patch('answerbot.get_wikipedia.WikipediaApi.get_page')
def test_search_success(mock_get_page, mock_get):
    api = WikipediaApi()
    mock_get_page.return_value = ContentRecord(None, ['Page retrieval history'])
    result = api.search("test_query")
    assert result.document is None
    assert "Wikipedia search results for query: 'test_query'" in result.retrieval_history[0]
    assert mock_get_page.called


def test_wikipedia_document_extraction():
    wiki_content = """
== Test Page ==
Main Title
Subtitle
This is the first paragraph. It's pretty long and contains a lot of text that should be split into chunks. Here's another sentence. And yet another one here. And more and more and more.
* Item 1 with the keyword
* Item 2
Another paragraph with the keyword.
== Section 2 ==
Paragraph with a new_keyword.
"""
    doc = WikipediaDocument(wiki_content, chunk_size=SMALL_CHUNK_SIZE)
    assert doc.text == wiki_content
    assert "This is the first paragraph." in doc.first_chunk()
    assert len(doc.first_chunk()) <= SMALL_CHUNK_SIZE
    assert doc.first_chunk().endswith(".") or doc.first_chunk().endswith("!") or doc.first_chunk().endswith("?")
    assert doc.section_titles() == ['Test Page', 'Section 2']
    assert "Item 1 with the keyword" in doc.lookup("keyword")
    assert "Paragraph with a new_keyword." in doc.lookup("new_keyword")
    assert doc.lookup("new_keyword").startswith("== Section 2 ==")
    assert doc.lookup("Section 2").startswith("== Section 2 ==")
    assert doc.lookup("nonexistent") is None
    wiki_content = "A (b)"
    doc = WikipediaDocument(wiki_content)
    assert wiki_content in doc.lookup("a")
    assert wiki_content in doc.lookup("A (b)")


import pytest
import requests
from unittest.mock import patch, Mock
from answerbot.get_wikipedia import WikipediaApi, WikipediaDocument, MarkdownDocument, ContentRecord

SMALL_CHUNK_SIZE = 72

@pytest.fixture
def wikipedia_api():
    return WikipediaApi(max_retries=3, chunk_size=500)

@patch('requests.get')
def test_successful_retrieval(mock_get, wikipedia_api):
    mock_get.return_value.json.return_value = {
        'query': {
            'pages': {
                '12345': {
                    'pageid': 12345,
                    'ns': 0,
                    'title': 'Python',
                    'revisions': [{'*': '<html>Sample content</html>'}]
                }
            }
        }
    }
    mock_get.return_value.status_code = 200

    record = wikipedia_api.get_page("Python")

    assert "Successfully retrieved 'Python' from Wikipedia." in record.retrieval_history
    assert record.document is not None
    assert 'Sample content' in record.document.content

@patch.object(WikipediaApi, 'get_page')
@patch('requests.get')
def test_search_with_results(mock_get, mock_get_page, wikipedia_api):
    mock_get.return_value.json.return_value = {
        'query': {
            'search': [{'title': 'Python (programming)'}]
        }
    }
    mock_get.return_value.status_code = 200

    mock_document = Mock(content="Sample content for Python")
    mock_get_page.return_value = ContentRecord(mock_document, ["Sample retrieval history"])

    record = wikipedia_api.search("Python")

    mock_get_page.assert_called_once_with("Python (programming)")
    assert "Wikipedia search results for query: 'Python' are: [[Python (programming)]]" in record.retrieval_history
    assert "Sample retrieval history" in record.retrieval_history
    assert record.document.content == "Sample content for Python"

@patch.object(WikipediaApi, 'get_page')
@patch('requests.get')
def test_search_no_results(mock_get, mock_get_page, wikipedia_api):
    mock_get.return_value.json.return_value = {'query': {'search': []}}
    mock_get.return_value.status_code = 200

    record = wikipedia_api.search("NonExistentTopic")

    mock_get_page.assert_not_called()
    assert "Wikipedia search results for query: 'NonExistentTopic' are: " in record.retrieval_history
    assert record.document is None

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

def test_markdown_document_extraction():
    wiki_content = """
## Test Page
### Main Title
#### Subtitle
This is the first paragraph. It's pretty long and contains a lot of text that should be split into chunks. Here's another sentence. And yet another one here. And more and more and more.
* Item 1 with the keyword
* Item 2
Another paragraph with the keyword.
## Section 2
Paragraph with a new_keyword.
"""
    doc = MarkdownDocument(wiki_content, chunk_size=SMALL_CHUNK_SIZE)
    assert doc.text == wiki_content
    assert "This is the first paragraph." in doc.first_chunk()
    assert len(doc.first_chunk()) <= SMALL_CHUNK_SIZE
    assert doc.section_titles() == ['## Test Page', '### Main Title', '#### Subtitle', '## Section 2']
    assert "Item 1 with the keyword" in doc.lookup("keyword")
    assert "Paragraph with a new_keyword." in doc.lookup("new_keyword")
    assert doc.lookup("new_keyword").startswith("## Section 2")
    assert doc.lookup("Section 2").startswith("## Section 2")
    assert doc.lookup("## Section 2").startswith("## Section 2")
    assert doc.lookup("nonexistent") is None
    wiki_content = "A (b)"
    doc = MarkdownDocument(wiki_content)
    assert wiki_content in doc.lookup("a")
    assert wiki_content in doc.lookup("A (b)")

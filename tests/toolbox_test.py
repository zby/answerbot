from unittest.mock import MagicMock, Mock
from answerbot.get_wikipedia import ContentRecord, Document
from answerbot.toolbox import WikipediaSearch
import pytest

@pytest.fixture
def wiki_search():
    wiki_api = MagicMock()
    return WikipediaSearch(wiki_api)

def test_process_with_unknown_function(wiki_search):
    with pytest.raises(Exception) as context:
        wiki_search.process('unknown_function', {'query': 'Python', 'reason': 'Test search'})

    assert 'Unknown function name: unknown_function' == str(context.value)

def test_search(wiki_search):
    mock_search_record = MagicMock(spec=ContentRecord)
    mock_document = Mock()
    mock_document.section_titles.return_value = ['test section']
    mock_document.first_chunk.return_value = 'test text'
    mock_search_record.document = mock_document
    mock_search_record.retrieval_history = ['test test']

    wiki_search.wiki_api.search.return_value = mock_search_record
    test_response = wiki_search.process('search', {'query': 'Python', 'reason': 'Test search'})
    assert test_response == wiki_search.retrieval_observations(mock_search_record)

def test_get(wiki_search):
    mock_search_record = MagicMock(spec=ContentRecord)
    mock_search_record.retrieval_history = ['test test']
    mock_document = Mock()
    mock_document.section_titles.return_value = ['test section']
    mock_document.first_chunk.return_value = 'test text'
    mock_search_record.document = mock_document

    wiki_search.wiki_api.get_page.return_value = mock_search_record
    test_response = wiki_search.process('get', {'title': 'Python', 'reason': 'Test get'})
    assert test_response == wiki_search.retrieval_observations(mock_search_record)

def test_lookup_with_no_document(wiki_search):
    function_args = {'keyword': 'Python', 'reason': 'Test lookup'}
    test_response = wiki_search.process('lookup', function_args)
    assert test_response == "No document defined, cannot lookup"

def test_lookup_with_document(wiki_search):
    mock_document = MagicMock(spec=Document)
    mock_document.lookup_results = ["Mock text"]
    mock_document.lookup.return_value = "Mock text"
    wiki_search.process('search', {'query': 'Python', 'reason': 'Test search'})
    wiki_search.document = mock_document
    function_args = {'keyword': 'Python', 'reason': 'Test lookup'}
    test_response = wiki_search.process('lookup', function_args)
    assert test_response == 'Keyword "Python" found on current page in 1 places. The first occurence:\nMock text'

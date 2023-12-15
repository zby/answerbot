import unittest
from unittest.mock import MagicMock, Mock
from answerbot.get_wikipedia import ContentRecord, Document
from answerbot.toolbox import WikipediaSearch


class TestWikipediaSearch(unittest.TestCase):
    def setUp(self):
        self.wiki_api = MagicMock()
        self.wiki_search = WikipediaSearch(self.wiki_api)

    def test_process_with_unknown_function(self):
        with self.assertRaises(Exception) as context:
            self.wiki_search.process('unknown_function', {'query': 'Python', 'reason': 'Test search'})

        self.assertEqual('Unknown function name: unknown_function', str(context.exception))

    def test_search(self):
        mock_search_record = MagicMock(spec=ContentRecord)
        mock_document = Mock()
        mock_document.section_titles.return_value = ['test section']
        mock_document.first_chunk.return_value = 'test text'
        mock_search_record.document = mock_document
        mock_search_record.retrieval_history = ['test test']

        self.wiki_api.search.return_value = mock_search_record
        test_response = self.wiki_search.process('search', {'query': 'Python', 'reason': 'Test search'})
        self.assertEqual(test_response, self.wiki_search.retrieval_observations(mock_search_record))

    def test_get(self):
        mock_search_record = MagicMock(spec=ContentRecord)
        mock_search_record.retrieval_history = ['test test']
        mock_document = Mock()
        mock_document.section_titles.return_value = ['test section']
        mock_document.first_chunk.return_value = 'test text'
        mock_search_record.document = mock_document

        self.wiki_api.get_page.return_value = mock_search_record
        test_response = self.wiki_search.process('get', {'title': 'Python', 'reason': 'Test get'})
        self.assertEqual(test_response, self.wiki_search.retrieval_observations(mock_search_record))
        self.wiki_api.search.return_value = mock_search_record
        test_response = self.wiki_search.process('search', {'query': 'Python', 'reason': 'Test search'})
        self.assertEqual(test_response, self.wiki_search.retrieval_observations(mock_search_record))

    def test_lookup_with_no_document(self):
        function_args = {'keyword': 'Python', 'reason': 'Test lookup'}
        test_response = self.wiki_search.process('lookup', function_args)
        self.assertEqual(test_response, "No document defined, cannot lookup")

    def test_lookup_with_document(self):
        mock_document = MagicMock(spec=Document)
        mock_document.lookup_results = ["Mock text"]
        mock_document.lookup.return_value = "Mock text"
        self.wiki_search.process('search', {'query': 'Python', 'reason': 'Test search'})
        self.wiki_search.document = mock_document
        function_args = {'keyword': 'Python', 'reason': 'Test lookup'}
        test_response = self.wiki_search.process('lookup', function_args)
        self.assertEqual(test_response, 'Keyword "Python" found on current page in 1 places. The first occurence:\nMock text')


if __name__ == '__main__':
    unittest.main()
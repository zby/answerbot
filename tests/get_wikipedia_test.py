import unittest
from unittest.mock import patch, Mock
from get_wikipedia import WikipediaApi, ContentRecord
import wikipedia

# this code is from gpt4
# it failed with testing the exceptions

class TestWikipediaApi(unittest.TestCase):
    def setUp(self):
        # Create an instance of WikipediaApi for testing
        self.wiki_api = WikipediaApi(max_retries=3)

    @patch('get_wikipedia.wikipedia')
    def test_get_page_success(self, mock_wikipedia):
        # Mock the behavior of wikipedia.page
        mock_page = Mock()
        mock_page.summary = "Mocked page summary"
        mock_wikipedia.page.return_value = mock_page

        # Test retrieving a page successfully
        title = "Python (programming language)"
        content_record = self.wiki_api.get_page(title)

        # Check that the page summary is correctly set in the ContentRecord
        self.assertEqual(content_record.page.summary, "Mocked page summary")

    @patch('get_wikipedia.wikipedia')
    def test_search_success(self, mock_wikipedia):
        # Mock the behavior of wikipedia.search
        mock_wikipedia.search.return_value = ['Machine learning', 'Other Result']

        # Mock the behavior of wikipedia.page for the first search result
        mock_page = Mock()
        mock_page.summary = "Mocked page summary"
        mock_wikipedia.page.return_value = mock_page

        # Test searching for 'Machine learning' and retrieving the first result
        search_query = "Machine learning"
        content_record = self.wiki_api.search(search_query)

        history = content_record.retrieval_history
        self.assertEqual(history[0], "Search results for query: 'Machine learning': 'Machine learning', 'Other Result'")
        self.assertEqual(history[1], "Successfully retrieved 'Machine learning' from Wikipedia.")
        self.assertEqual(len(history), 2)

        # Check that the page summary is correctly set in the ContentRecord
        self.assertEqual(content_record.page.summary, "Mocked page summary")

    @patch('get_wikipedia.wikipedia')
    def test_search_no_results(self, mock_wikipedia):
        # Mock the behavior of wikipedia.search to return no results
        mock_wikipedia.search.return_value = []

        # Test searching for a query with no results
        search_query = "Nonexistent Query"
        content_record = self.wiki_api.search(search_query)

        # Check that the retrieval history indicates no search results
        self.assertIn("No search results found for query:", content_record.retrieval_history[0])

if __name__ == '__main__':
    unittest.main()

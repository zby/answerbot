import unittest
import wikipedia
import requests
from unittest.mock import patch, Mock
from get_wikipedia import WikipediaApi, WikipediaDocument, ContentRecord

SMALL_CHUNK_SIZE = 70

class TestWikipediaApi(unittest.TestCase):
    def setUp(self):
        self.api = WikipediaApi(max_retries=3, chunk_size=500)  # Adjust chunk size as needed

    @patch('requests.get')
    def test_successful_retrieval(self, mock_get):
        # Mock the API response for successful page retrieval
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

        record = self.api.get_page("Python")

        self.assertIn("Successfully retrieved 'Python' from Wikipedia.", record.retrieval_history)
        self.assertIsNotNone(record.document)
        self.assertIn('Sample content', record.document.content)


    @patch.object(WikipediaApi, 'get_page')
    @patch('requests.get')
    def test_search_with_results(self, mock_get, mock_get_page):
        # Mock the API response for search
        mock_get.return_value.json.return_value = {
            'query': {
                'search': [{
                    'title': 'Python (programming)'
                }]
            }
        }
        mock_get.return_value.status_code = 200

        # Mock the behavior of get_page to return a sample ContentRecord
        mock_document = Mock(content="Sample content for Python")
        mock_get_page.return_value = ContentRecord(mock_document, ["Sample retrieval history"])

        record = self.api.search("Python")

        # Verify that get_page was called with the expected argument
        mock_get_page.assert_called_once_with("Python (programming)")

        # Check if the search history and get_page retrieval history are both recorded
        self.assertIn("Wikipedia search results for query: 'Python' are: [[Python (programming)]]", record.retrieval_history)
        self.assertIn("Sample retrieval history", record.retrieval_history)
        self.assertEqual(record.document.content, "Sample content for Python")


    @patch.object(WikipediaApi, 'get_page')
    @patch('requests.get')
    def test_search_no_results(self, mock_get, mock_get_page):
        # Mock the API response for search with no results
        mock_get.return_value.json.return_value = {
            'query': {
                'search': []  # Empty list to simulate no search results
            }
        }
        mock_get.return_value.status_code = 200

        record = self.api.search("NonExistentTopic")

        # Verify that get_page was not called
        mock_get_page.assert_not_called()

        # Check if the search history indicates no results
        self.assertIn("Wikipedia search results for query: 'NonExistentTopic' are: ", record.retrieval_history)
        self.assertIsNone(record.document)

class TestWikipediaDocument(unittest.TestCase):

    def test_wikipedia_document_extraction(self):
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
        # print(doc.text)
        self.assertEqual(doc.text, wiki_content)

        # Test the first chunk
        self.assertIn("This is the first paragraph.", doc.first_chunk())
        # Making sure the first_chunk doesn't exceed the smaller chunk_size (not defined here, should be based on your implementation)
        self.assertLessEqual(len(doc.first_chunk()), SMALL_CHUNK_SIZE)
        # Also checking if the chunk ends with a full sentence (i.e., not in between)
        self.assertTrue(doc.first_chunk().endswith(".") or doc.first_chunk().endswith(
            "!") or doc.first_chunk().endswith("?"))

        # Test section titles
        expected_section_titles = ['Test Page', 'Section 2']
        self.assertEqual(doc.section_titles(), expected_section_titles)

        # Test lookup
        self.assertIn("Item 1 with the keyword", doc.lookup("keyword"))
        self.assertIn("Paragraph with a new_keyword.", doc.lookup("new_keyword"))
        self.assertTrue(doc.lookup("new_keyword").startswith("== Section 2 =="))
        self.assertTrue(doc.lookup("Section 2").startswith("== Section 2 =="))
        print(doc.lookup("Section 2"))
        self.assertIsNone(doc.lookup("nonexistent"))

        # more lookup tests
        wiki_content = "A (b)"
        doc = WikipediaDocument(wiki_content)

        self.assertIn(wiki_content, doc.lookup("a"))  # should be case insensitive
        self.assertIn(wiki_content, doc.lookup("A (b)")) # should not treat keyword as regex


if __name__ == '__main__':
    unittest.main()

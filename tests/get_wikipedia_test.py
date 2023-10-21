import unittest
from unittest.mock import patch, Mock
from get_wikipedia import WikipediaApi, WikipediaDocument

SMALL_CHUNK_SIZE = 70

class TestWikipediaApi(unittest.TestCase):
    # this code is from gpt4
    # it failed with testing the exceptions
    def setUp(self):
        # Create an instance of WikipediaApi for testing
        self.wiki_api = WikipediaApi(max_retries=3)

    @patch('get_wikipedia.wikipedia')
    def test_get_page_success(self, mock_wikipedia):
        # Mock the behavior of wikipedia.page
        mock_page = Mock()
        mock_page.content = "Mocked page content"
        mock_wikipedia.page.return_value = mock_page

        # Test retrieving a page successfully
        title = "Python (programming language)"
        content_record = self.wiki_api.get_page(title)

        # Check content
        self.assertEqual(content_record.document.content, "Mocked page content")

    @patch('get_wikipedia.wikipedia')
    def test_search_success(self, mock_wikipedia):
        # Mock the behavior of wikipedia.search
        mock_wikipedia.search.return_value = ['Machine learning', 'Other Result']

        # Mock the behavior of wikipedia.page for the first search result
        mock_results = ['Machine learning', 'Other Result']
        mock_wikipedia.search.return_value = mock_results

        mock_page = Mock()
        mock_page.summary = "Mocked page summary"
        mock_page.content = "Mocked page content"
        mock_page.title = "Mocked page title"
        mock_wikipedia.page.return_value = mock_page

        # Test searching for 'Machine learning' and retrieving the first result
        search_query = "Machine learning"
        content_record = self.wiki_api.search(search_query)

        history = content_record.retrieval_history
        self.assertEqual(history[0], "Wikipedia search results for query: 'Machine learning' is: 'Machine learning', 'Other Result'")
        self.assertEqual(history[1], "Successfully retrieved 'Machine learning' from Wikipedia.")
        self.assertEqual(len(history), 2)

        # Check that the page summary is correctly set in the ContentRecord
        self.assertEqual(content_record.document.summary, "Mocked page summary")

    @patch('get_wikipedia.wikipedia')
    def test_search_no_results(self, mock_wikipedia):
        # Mock the behavior of wikipedia.search to return no results
        mock_wikipedia.search.return_value = []

        # Test searching for a query with no results
        search_query = "Nonexistent Query"
        content_record = self.wiki_api.search(search_query)

        # Check that the retrieval history indicates no search results
        self.assertIn("No search results found for query:", content_record.retrieval_history[0])


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
        self.assertEqual(doc.text, ' '.join(wiki_content.split()))

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
        self.assertIsNone(doc.lookup("nonexistent"))

        # more lookup tests
        wiki_content = "A (b)"
        doc = WikipediaDocument(wiki_content)

        self.assertIn(wiki_content, doc.lookup("a"))  # should be case insensitive
        self.assertIn(wiki_content, doc.lookup("A (b)")) # should not treat keyword as regex


if __name__ == '__main__':
    unittest.main()

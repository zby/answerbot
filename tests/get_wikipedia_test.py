import unittest
import wikipedia
from unittest.mock import patch, Mock
from get_wikipedia import WikipediaApi, WikipediaDocument, ContentRecord

SMALL_CHUNK_SIZE = 70

class TestWikipediaApi(unittest.TestCase):
    # this code is from gpt4
    # it failed with testing the exceptions
    def setUp(self):
        # Create an instance of WikipediaApi for testing
        self.wiki_api = WikipediaApi(max_retries=3)
        self.api = WikipediaApi(max_retries=3, chunk_size=SMALL_CHUNK_SIZE)

    @patch("wikipedia.page")
    def test_successful_retrieval(self, mock_page):
        mock_content = "Sample content"
        mock_page.return_value = Mock(content=mock_content, links=[])

        record = self.api.get_page("Python")

        self.assertTrue("Successfully retrieved 'Python' from Wikipedia." in record.retrieval_history)
        self.assertIsNotNone(record.document)
        self.assertEqual(record.document.content, mock_content)  # Thi
    @patch("wikipedia.page")
    def test_disambiguation_error(self, mock_page):
        mock_page.side_effect = wikipedia.DisambiguationError("Python", ["Python (programming)", "Python (snake)"])
        record = self.api.get_page("Python")
        self.assertTrue("Retrieved disambiguation page for 'Python'. Options: Python (programming), Python (snake)" in record.retrieval_history)

    @patch("wikipedia.page")
    def test_redirect_error(self, mock_page):
        mock_page.side_effect = wikipedia.RedirectError("Python (programming)")
        record = self.api.get_page("Python")
        self.assertTrue("Python redirects to Python (programming)" in record.retrieval_history)

    @patch("wikipedia.page")
    def test_page_error(self, mock_page):
        mock_page.side_effect = wikipedia.PageError("Python")
        record = self.api.get_page("Python")
        self.assertTrue("Page 'Python' does not exist." in record.retrieval_history)


    @patch.object(WikipediaApi, 'get_page')
    @patch("wikipedia.search")
    def test_search_with_results(self, mock_search, mock_get_page):
        # Mock the behavior of the wikipedia search to return a result
        mock_search.return_value = ["Python (programming)"]

        # Mock the behavior of get_page to return a sample ContentRecord
        mock_document = Mock(content="Sample content for Python")
        mock_get_page.return_value = ContentRecord(mock_document, ["Sample retrieval history"])

        record = self.api.search("Python")

        # Verify that get_page was called with the expected argument
        mock_get_page.assert_called_once_with("Python (programming)")

        # Check if the search history and get_page retrieval history are both recorded
        self.assertIn("Wikipedia search results for query: 'Python' is: [[Python (programming)]]", record.retrieval_history)
        self.assertIn("Sample retrieval history", record.retrieval_history)
        self.assertEqual(record.document.content, "Sample content for Python")


    @patch.object(WikipediaApi, 'get_page')
    @patch("wikipedia.search")
    def test_search_no_results(self, mock_search, mock_get_page):
        # Mock the behavior of the wikipedia search to return no results
        mock_search.return_value = []

        record = self.api.search("NonExistentTopic")

        # Verify that get_page was not called
        mock_get_page.assert_not_called()

        # Check if the search history indicates no results
        self.assertIn("Wikipedia search results for query: 'NonExistentTopic' is: ", record.retrieval_history)
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
